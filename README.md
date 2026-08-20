# AWS Durable Execution SDK for Rust

A durable function is a Lambda function whose progress the service
checkpoints as it runs. Each invocation ends when the handler suspends on a
timer, an external signal, or the function timeout. The service then invokes
the function again, and this SDK replays the recorded results instead of
re-running the work that already completed. An orchestration that spans
minutes or a month therefore fits in one ordinary async Rust function, and
once the service records a step's result, replay returns that result
without re-running the body.

## Status

Preview. The `alpha` branch carries the current code. Steps, waits, invokes,
callbacks, child contexts, map, parallel, and the four combinators all work
against the live service, and the API may still change. `make check` runs the
quality gate: formatting, clippy, tests and doctests, docs, and the
dependency policy.

The crate is not on crates.io yet, so depend on the repository directly. It
requires Rust 1.94.1 or newer and edition 2024.

```toml
[dependencies]
aws-durable-execution-sdk-rust = { git = "https://github.com/aws/aws-durable-execution-sdk-rust", branch = "alpha" }
lambda_runtime = "1"
serde_json = "1"
tokio = { version = "1", features = ["macros"] }
```

## Your first durable function

Create a binary crate named `first_durable_function` with the dependencies
above and put this in `src/main.rs`. The handler runs a step, suspends on a
two second timer, then runs a second step that reads the first result.

```rust
use std::time::Duration;

use aws_durable_execution_sdk_rust as durable;

async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    let name = ctx
        .step(|_| async { Ok("world".to_owned()) })
        .name("fetch-name")
        .await?;

    ctx.wait(Duration::from_secs(2)).name("cooldown").await?;

    let greeting = ctx
        .step(move |_| async move { Ok(format!("hello, {name}")) })
        .name("format")
        .await?;

    Ok(greeting)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
```

Build it with [cargo-lambda](https://www.cargo-lambda.info/), which the
Lambda Developer Guide documents for
[packaging Rust functions](https://docs.aws.amazon.com/lambda/latest/dg/rust-package.html).
Install it with `pip3 install cargo-lambda` or `cargo install cargo-lambda`.

```sh
cargo lambda build --release --x86-64
```

Use cargo-lambda rather than a plain `cargo build`. The `provided.al2023`
runtime ships glibc 2.34, so a binary that links against a newer glibc dies
at startup with `/lib64/libc.so.6: version 'GLIBC_2.39' not found`.
cargo-lambda builds through cargo-zigbuild, which pins the linked glibc
version, producing a runtime-compatible artifact on any host.

The build writes `target/lambda/first_durable_function/bootstrap`. Point a SAM
template at that directory and give the function a `DurableConfig`, which is
what makes the service checkpoint it.

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: First durable function

Parameters:
  ExecutionRoleArn:
    Type: String
    Description: IAM role the function assumes

Resources:
  FirstDurableFunction:
    Type: AWS::Serverless::Function
    Properties:
      FunctionName: first-durable-function
      CodeUri: target/lambda/first_durable_function/
      Handler: bootstrap
      Runtime: provided.al2023
      Architectures: [x86_64]
      Timeout: 60
      MemorySize: 128
      Role: !Ref ExecutionRoleArn
      DurableConfig:
        RetentionPeriodInDays: 7
        ExecutionTimeout: 300
```

The template takes an existing execution role rather than creating one. That
role needs CloudWatch Logs write access plus the two durable execution
permissions the SDK calls, `lambda:CheckpointDurableExecution` and
`lambda:GetDurableExecutionState`.

```sh
sam deploy --template-file template.yaml \
    --stack-name first-durable-function \
    --resolve-s3 --region us-west-2 \
    --parameter-overrides ExecutionRoleArn=arn:aws:iam::111122223333:role/durable-lambda-role
```

Invoking a durable function starts an execution and returns immediately with
its ARN. The invocation response does not carry the handler's return value,
because the execution outlives that first invocation.

```sh
aws lambda invoke --function-name first-durable-function \
    --qualifier '$LATEST' --payload '{}' \
    --cli-binary-format raw-in-base64-out response.json
```

The CLI writes the handler payload to `response.json` and prints the invoke
metadata to stdout. Take `DurableExecutionArn` from that metadata and poll the
execution until it leaves `RUNNING`. It reaches `SUCCEEDED` a couple of seconds
after the timer fires, with `"hello, world"` as its result.

```sh
aws lambda get-durable-execution \
    --durable-execution-arn <execution-arn> \
    --query '[Status,Result]' --output text
```

## The handler and the context

A durable handler is an async function of two arguments that returns a
`Result`.

```rust
async fn handler(
    event: MyEvent,
    ctx: durable::DurableContext,
) -> Result<MyOutput, durable::BoxError>
```

`durable::run` registers it with the Lambda runtime and drives one execution
per invocation. The event type needs `Deserialize` and the output type needs
`Serialize`; both are yours to choose. `BoxError` is
`Box<dyn Error + Send + Sync>`, so `?` accepts any error type without a
conversion. Use `durable::wrap` instead of `durable::run` when you want to
supply `Options` or add your own middleware around the service function.

A handler failure does not fail the Lambda invocation. The SDK reports it
inside a successful invocation response, as a `FAILED` envelope the durable
execution service reads, which is what moves the execution to the `FAILED`
status. Lambda-level error signals therefore stay quiet: the CloudWatch
`Errors` metric does not fire, dead-letter queues and `OnFailure`
destinations do not trigger, and X-Ray does not mark the trace as an error.
Monitor and alarm on the durable execution status instead, via
`get-durable-execution` or `list-durable-executions-by-function`.

`DurableContext` is a cheap-to-clone handle backed by an `Arc`, and it is
`Send + Sync`, so clone it across async boundaries at will. Besides the
operations below it exposes `execution_arn()`, `lambda_context()` for the
request ID and deadline, and `is_replaying()`, which tells you whether the
current invocation is replaying recorded results.

Every operation method returns a builder. Builders implement `IntoFuture`, so
awaiting the builder runs the operation; there is no separate build step.
`.name("...")` labels the operation in the execution history. `.future()`
converts the builder into a `DurableFuture<O>` that the combinators accept.
`.spawn()` starts the operation immediately and returns the same
`DurableFuture<O>`, which lets you overlap independent work and join it with
`tokio::join!`.

```rust
let wait = ctx.wait(Duration::from_secs(5)).name("timer").spawn();
let work = ctx.step(|_| async { Ok(42_i32) }).name("compute").spawn();

let (timer, result) = tokio::join!(wait, work);
timer?;
let value = result?;
```

### What determinism requires

The SDK claims each operation's ID synchronously at the call site, so replay
depends on your handler creating the same operations in the same order every
time. Keep the control flow that decides which operations to create a pure
function of the event and of results the SDK already recorded. Iterating a
`HashMap` or `HashSet` to create operations breaks that, because the iteration
order changes between runs; sort into a `Vec` first.

Put nondeterminism inside a step body instead of between operations. A step
body may read the clock, generate a random ID, or call a service, because only
its recorded result takes part in replay. `StepContext`, the argument a step
body receives, exposes just `attempt()` and no operation methods, so calling a
durable operation inside a step body fails to compile rather than failing at
run time.

Two more constraints follow from how suspension works. Use `ctx.race` or
`ctx.select_ok` rather than `tokio::select!` over durable futures, so the SDK
records which branch won. And when an execution suspends, the SDK drops your
handler future, so never make correctness between operations depend on `Drop`
order.

## Operations

| Operation | Behavior |
| --- | --- |
| `ctx.step(f)` | Runs `f` and checkpoints its result. On replay, returns the recorded result without re-running `f`. |
| `ctx.wait(duration)` | Suspends the execution for `duration`. |
| `ctx.invoke(function_id, input)` | Calls another durable function and waits for its result. |
| `ctx.run_in_child_context(f)` | Runs `f` against a child context with its own operation namespace. |
| `ctx.with_retry(f)` | Runs `f` against a child context and retries the whole block as a unit, with a fresh operation namespace per attempt. |
| `ctx.wait_for_condition(check, state)` | Polls `check`, carrying state between attempts, until the wait strategy completes. |
| `ctx.create_callback()` | Mints a callback ID now and resolves when an external system completes it. |
| `ctx.wait_for_callback(submitter)` | Mints the ID, hands it to `submitter`, and waits for the payload. |
| `ctx.map(items, f)` | Runs `f` per item, each in its own child context. |
| `ctx.parallel(branches)` | Runs named branches concurrently. |
| `ctx.try_join_all(futures)` | Collects every result, failing on the first error. |
| `ctx.join_all(futures)` | Collects every outcome as `Settled<O>`, never failing fast. |
| `ctx.select_ok(futures)` | Returns the first success and drops the losers. |
| `ctx.race(futures)` | Returns the first outcome to settle, success or failure. |

### step

A step is the unit of checkpointing. Its output needs
`Serialize + DeserializeOwned`, which is how the SDK records and restores it.

Once a step succeeds and the service records its result, replay returns that
result without re-running the body. Between the moment the SDK starts a step
and the moment the service records the outcome, an interruption (function
timeout, runtime crash) can occur. Under the default `AtLeastOncePerRetry`
semantics the SDK re-executes the body on resume, so the body may run more
than once per retry attempt if an interruption occurs in that window. Design
step bodies to be idempotent, or use `StepSemantics::AtMostOncePerRetry` to
treat an interrupted attempt as a failure rather than re-executing it.

```rust
let value = ctx
    .step(|_| async { Ok("step completed".to_owned()) })
    .await?;
```

Add `.retry_strategy(...)` to turn transient failures into retries. The
strategy takes the error and the 1-based attempt number and returns
`RetryDecision::Retry { delay }` or `RetryDecision::Stop`. The execution
suspends for the delay between attempts, so a retrying step does not hold the
invocation open.

```rust
let result = ctx
    .step(|step_ctx| async move {
        let attempt = step_ctx.attempt();
        if attempt < 3 {
            return Err(format!("transient failure on attempt {attempt}").into());
        }
        Ok(format!("succeeded on attempt {attempt}"))
    })
    .name("flaky-call")
    .retry_strategy(|_err, attempt| {
        if attempt >= 3 {
            RetryDecision::Stop
        } else {
            RetryDecision::Retry {
                delay: Duration::from_secs(u64::from(attempt)),
            }
        }
    })
    .semantics(StepSemantics::AtMostOncePerRetry)
    .await?;
```

`StepSemantics::AtLeastOncePerRetry` is the default: the SDK re-executes the
body on resume if the previous attempt started but recorded no outcome.
`AtMostOncePerRetry` treats that interruption as a failure and consults the
retry strategy, so a non-idempotent body never runs twice for the same
attempt.

### wait

```rust
ctx.wait(Duration::from_secs(60)).name("cooldown").await?;
```

### invoke

`invoke` calls another durable function and resolves with its result. The
output type parameter comes first so you can turbofish it.

```rust
let receipt = ctx
    .invoke::<serde_json::Value, _>(&target_function_name, event)
    .name("delegate")
    .await?;
```

### wait_for_condition

`wait_for_condition` runs a check repeatedly, carrying checkpointed state
between attempts, and suspends between polls. `wait_strategy_fn` receives the
current state and the attempt number and returns `WaitDecision::complete()`
or `WaitDecision::continue_with(delay)`.

```rust
let count = ctx
    .wait_for_condition(|_ctx, state: i32| async move { Ok(state + 1) }, 0)
    .wait_strategy_fn(move |state: i32, _attempt| {
        if state >= threshold {
            WaitDecision::complete()
        } else {
            WaitDecision::continue_with(Duration::from_secs(1))
        }
    })
    .name("poll-until-ready")
    .await?;
```

Pass a `WaitStrategy` to `.wait_strategy(...)` instead when you want
configurable exponential backoff between polls. `WaitStrategy` controls
only the delay timing; it always returns `WaitDecision::continue_with(...)`
and never completes the operation on its own. The condition check itself
must drive the state toward a value that your `wait_strategy_fn` recognizes
as done, or use a `wait_strategy_fn` that stops after a maximum attempt
count. Build a `WaitStrategy` with `WaitStrategy::builder()`, setting
`initial_delay`, `max_delay`, and `backoff_factor`.

### callbacks

`create_callback` mints an ID and returns a `Callback` immediately, so you can
hand the ID to an external system and await the payload separately. An
external caller completes it with the
`SendDurableExecutionCallbackSuccess` API.

```rust
let cb = ctx.create_callback::<String>().name("approval").await?;
tracing::info!(callback_id = %cb.id(), "awaiting external completion");
let approval = cb.result().await?;
```

`wait_for_callback` collapses those two halves into one operation: it mints
the ID, passes it to your submitter closure, and resolves with the payload.
The submitter receives the ID as an owned `String`, so move it straight into
the async block.

```rust
let approval: String = ctx
    .wait_for_callback::<String, _, _>(|_step_ctx, callback_id| async move {
        publish_approval_request(callback_id).await?;
        Ok(())
    })
    .name("await-approval")
    .await?;
```

Both builders take `.timeout(...)` and `.heartbeat(...)`.
`wait_for_callback` also takes `.submitter_retry(...)`, which retries the
submitter with the same retry strategy shape a step uses.

### child contexts

A child context gets its own operation namespace, which keeps a subroutine's
operation IDs independent of the caller's.

```rust
let greeting = ctx
    .run_in_child_context(|child| async move {
        let name = child
            .step(|_| async { Ok("world".to_owned()) })
            .name("fetch-name")
            .await?;
        Ok(format!("hello, {name}"))
    })
    .name("greet")
    .await?;
```

### with_retry

`with_retry` runs a closure against a child context and applies a retry
strategy to the closure's overall outcome, so a multi-operation block
retries as a unit. Each attempt gets a fresh child operation namespace: a
failed attempt's recorded operations are never replayed into the next
attempt, so every operation in the block re-runs on retry. Delays between
attempts suspend the execution, exactly as step retries do, and the retry
progress is derived from checkpointed results, so it survives suspension.
The closure is `Fn` rather than `FnOnce` because the SDK calls it once per
attempt.

```rust
let receipt = ctx
    .with_retry(|child| async move {
        let quote = child
            .step(|_| async { Ok(fetch_quote().await?) })
            .name("fetch-quote")
            .await?;
        let receipt = child
            .step(move |_| async move { Ok(book(quote).await?) })
            .name("book")
            .await?;
        Ok(receipt)
    })
    .name("quote-and-book")
    .retry_strategy(|_err, attempt| {
        if attempt >= 3 {
            RetryDecision::Stop
        } else {
            RetryDecision::Retry { delay: Duration::from_secs(5) }
        }
    })
    .await?;
```

`.retry_strategy_config(...)` accepts a `RetryStrategyConfig` instead of a
closure, and without either the step default applies (6 total attempts with
exponential backoff). When retries exhaust, the operation fails with the
attempt count and the last attempt's error.

### map and parallel

`map` applies one closure to every item, giving each item its own child
context. Items need `Serialize + DeserializeOwned` because the SDK records
them.

```rust
let results = ctx
    .map(items, |child, item, idx| async move {
        let out = child
            .step(move |_| async move { Ok(format!("item-{idx}:{item}")) })
            .name("process")
            .await?;
        Ok(out)
    })
    .name("process-all")
    .max_concurrency(4)
    .await?;
```

`parallel` runs a fixed set of named branches that share an output type.

```rust
let branches: Vec<Branch<u32>> = vec![
    Branch::new("double", |child| async move {
        let base = child.step(|_| async { Ok(21u32) }).name("compute").await?;
        Ok(base * 2)
    }),
    Branch::new("wait-then-value", |child| async move {
        child.wait(Duration::from_secs(1)).name("cooldown").await?;
        Ok(100u32)
    }),
];

let results = ctx.parallel(branches).name("fan-out").await?;
```

Both accept `.max_concurrency(...)` and `.completion(...)`. A
`CompletionConfig` ends the batch early: `with_min_successful(n)` stops once
`n` items succeed, `with_tolerated_failure_count(0)` fails fast on the first
error, and `CompletionConfig::builder()` combines thresholds so the first one
to fire wins; its `build()` validates the combination and rejects a
misconfiguration (such as a percentage above 100) at construction time. The
builder's `.completion_predicate(...)` adds a custom trigger: a function of
the running batch statistics (succeeded count, failed count, total items,
and the settled outcomes so far) that returns `true` to end the batch with
the `PredicateMatched` reason. The predicate must be a deterministic, pure
function of the statistics it receives — the SDK evaluates it only on state
derived from recorded results, and anything else (clock, randomness, outside
state) can make replay diverge. To keep those statistics reproducible, item
outcomes feed them strictly in input order (item `i` enters only once items
`0..i` have all settled), whatever order the items actually finished in. It
composes with the fixed thresholds under the same first-trigger-wins rule,
checked after them.
`MapBuilder` and `ParallelBuilder` also expose `.await_batch()`:
await it instead of the builder to get a `BatchResult<O>` that reports each
item's status and why the batch ended. `BatchResult::status()` returns a
`BatchStatus` enum, and `BatchResult::errors()` returns `BatchError` entries
that tie each failure to the index and name of the item that produced it,
along with the error's type identifier and message.

### combinators

The four combinators take `DurableFuture` values, which `.future()` and
`.spawn()` produce, and record the combined outcome as one operation.

`try_join_all` collects every result, propagating the first error:

```rust
let a = ctx.step(|_| async { Ok(1u32) }).name("a").future();
let b = ctx.step(|_| async { Ok(2u32) }).name("b").future();
let c = ctx.step(|_| async { Ok(3u32) }).name("c").future();

let results: Vec<u32> = ctx.try_join_all([a, b, c]).name("gather").await?;
```

`join_all` returns `Vec<Settled<O>>` so you can inspect failures individually
without failing fast:

```rust
let good = ctx.step(|_| async { Ok(1u32) }).name("good").future();
let bad = ctx
    .step(|_| async { Err::<u32, durable::BoxError>("boom".into()) })
    .name("bad")
    .retry_strategy(|_err, _attempt| durable::RetryDecision::Stop)
    .future();

let settled: Vec<durable::Settled<u32>> = ctx.join_all([good, bad]).name("collect").await?;
for outcome in &settled {
    match outcome {
        durable::Settled::Fulfilled(value) => tracing::info!("ok: {value}"),
        durable::Settled::Rejected(err) => tracing::info!("err: {err}"),
        _ => {}
    }
}
```

`select_ok` returns the first success, dropping losers:

```rust
let primary = ctx
    .step(|_| async { Err::<u32, durable::BoxError>("unavailable".into()) })
    .name("primary")
    .retry_strategy(|_err, _attempt| durable::RetryDecision::Stop)
    .future();
let fallback = ctx.step(|_| async { Ok(42u32) }).name("fallback").future();

let winner: u32 = ctx.select_ok([primary, fallback]).name("first-ok").await?;
```

`race` returns the first outcome to settle, whether it succeeded or failed:

```rust
let first = ctx.step(|_| async { Ok("first".to_owned()) }).name("first").future();
let second = ctx.step(|_| async { Ok("second".to_owned()) }).name("second").future();

let winner: String = ctx.race([first, second]).name("fastest").await?;
```

## Custom serialization

The SDK stores operation payloads as JSON by default. Implement
`Serdes<T>` to change that. The trait is generic over the operation's
actual Rust type and asynchronous: `serialize` receives the **owned typed
value** and returns the `String` to store on the wire; `deserialize` takes
that wire string back to the typed value. There is no intermediate
representation — a custom format sees struct field declaration order,
`i128` values outside the `i64`/`u64` ranges, and everything else the real
type carries.

The SDK awaits the future a serdes returns directly on the executor
thread. Cheap in-memory transforms (like the default `JsonSerdes`) run
inline in the returned future; an implementation that blocks — filesystem
I/O, long-running synchronous work — must move that work into
`tokio::task::spawn_blocking` itself, the way `FileSystemSerdes` does.
`async fn` in the implementation satisfies the trait's `impl Future`
return type:

```rust
struct Base64JsonSerdes;

impl<T> Serdes<T> for Base64JsonSerdes
where
    T: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
{
    async fn serialize(
        &self,
        value: T,
        _context: SerdesContext,
    ) -> Result<String, durable::BoxError> {
        use std::io::Write;
        let json = serde_json::to_vec(&value)?;
        let mut buf = Vec::new();
        let engine = base64::engine::general_purpose::STANDARD;
        {
            let mut encoder =
                base64::write::EncoderWriter::new(&mut buf, &engine);
            encoder.write_all(&json)?;
            encoder.finish()?;
        }
        Ok(String::from_utf8(buf)?)
    }

    async fn deserialize(
        &self,
        wire: String,
        _context: SerdesContext,
    ) -> Result<T, durable::BoxError> {
        use std::io::Read;
        let engine = base64::engine::general_purpose::STANDARD;
        let mut decoder =
            base64::read::DecoderReader::new(wire.as_bytes(), &engine);
        let mut json = Vec::new();
        decoder.read_to_end(&mut json)?;
        Ok(serde_json::from_slice(&json)?)
    }
}
```

A **type-agnostic** format like the one above uses a blanket
`impl<T> Serdes<T>` and attaches to any operation. A **type-specific**
format implements `Serdes<ConcreteType>` directly — `impl Serdes<Order>
for OrderSerdes` — and gains compile-time pairing: attaching it to an
operation that produces another type fails to compile.

Serdes are configured **per operation**, with `.serdes(...)` on
`StepBuilder`, `InvokeBuilder`, `ChildBuilder`, `WithRetryBuilder`,
`WaitForConditionBuilder`, `CreateCallbackBuilder`,
`WaitForCallbackBuilder`, `MapBuilder`, or `ParallelBuilder`. Each builder
carries its serdes as a generic type parameter defaulting to `JsonSerdes`,
and `.serdes(...)` swaps that parameter; `.await` still produces a
`DurableFuture<O>` regardless of the serdes type, so futures configured
with different serdes implementations coexist in one combinator input
collection. `WaitBuilder` and the combinator builders carry no serdes
method because their payloads are structural, not user-typed.

There is no execution-wide serdes slot: a single trait-object slot cannot
represent `Serdes<T>` for every operation output type without erasing the
value again. To share one instance across a handler, create an `Arc<S>`
and clone it into each operation — `Arc<S>` forwards to `S`:

```rust
let shared = std::sync::Arc::new(Base64JsonSerdes);
let a: String = ctx
    .step(|_| async { Ok("a".to_owned()) })
    .serdes(std::sync::Arc::clone(&shared))
    .await?;
let b: u32 = ctx.step(|_| async { Ok(7_u32) }).serdes(shared).await?;
```

Operations transfer the owned value to `serialize` and return the value
`deserialize` reconstructs from the stored wire string, so live execution
and replay observe identical values. Ownership transfer is also what keeps
operation outputs at `Send` (not `Sync`): the serdes can move the whole
call into a `'static` blocking task without borrowing across an `.await`.

`FileSystemSerdes` stores payloads on a durable shared filesystem (Amazon EFS
or S3 Files mounted to Lambda), resolving a deterministic path from the
`SerdesContext` so replay finds the same file. Each `serialize` or
`deserialize` call runs its complete implementation — JSON rendering or
parsing, path resolution, and the file I/O — inside one
`tokio::task::spawn_blocking` task, so the executor thread never touches
the filesystem. Do not use it with Lambda's ephemeral `/tmp`: that storage
is local to a single execution environment and does not persist across
invocations. Mount EFS or S3 Files and point `FileSystemSerdes` at the
mount path.

```rust
use durable::serdes::{FileSystemSerdes, FileSystemSerdesConfig, FileSystemSerdesMode};

let serdes = FileSystemSerdes::with_config(
    "/mnt/efs",
    FileSystemSerdesConfig::builder()
        .storage_mode(FileSystemSerdesMode::Overflow)
        .build(),
);
```

## Testing a handler locally

Enable the `test-util` feature to get `LocalRunner`, which drives a handler
through as many simulated invocations as the execution needs, in process and
without AWS.

```rust
let result = LocalRunner::new()
    .run(
        |n: u32, ctx: durable::DurableContext| async move {
            let v = ctx.step(move |_| async move { Ok(n + 1) }).await?;
            Ok::<_, durable::BoxError>(v)
        },
        41_u32,
    )
    .await;
assert_eq!(result.output(), Some(&42));
```

`TestResult` also reports the invocation count and every recorded operation,
so a test can assert that a step ran once across replays.
`.callback_success(&value)` queues a payload for the next callback the handler
awaits, and `.callback_timeout()` times one out.

## Examples

`examples/` holds one deployable function per pattern: `basics/` for steps,
waits, and retries, `coordination/` for child contexts, map, parallel, and the
combinators, and `external/` for invoke, callbacks, `wait_for_condition`,
serdes, large payloads, and replay-suppressed logging. Each one is a Lambda
function on `provided.al2023`, and cargo-lambda builds each example the same
way as the quick start above. Each reads as a single workload rather than a
demo harness.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, the quality
gate, and how to run the test suites.

## License

Apache-2.0. See [LICENSE](LICENSE).
