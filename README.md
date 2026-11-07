# AWS Durable Execution SDK for Rust

A durable function is a Lambda function whose progress the service
checkpoints as it runs. Each invocation ends when the handler suspends on a
timer, an external signal, or the function timeout. The service then invokes
the function again, and this SDK replays the recorded results instead of
re-running the work that already completed. An orchestration that spans
minutes or a month therefore fits in one ordinary async Rust function, and
once the service records a step's result, replay returns that result
without re-running the body.

> [!WARNING]
> This is an experimental preview, not intended for production use. The API
> may change without notice, and the final version may look different from
> what you see today.

The SDK requires Rust 1.94.1 or newer. The SDK crate uses edition 2024, but
a crate that depends on it can use edition 2018 or later.

```toml
[dependencies]
aws-durable-execution-sdk = { git = "https://github.com/aws/aws-durable-execution-sdk-rust" }
lambda_runtime = "1"
serde_json = "1"
tokio = { version = "1", features = ["macros"] }
```

## Your first durable function

Create a binary crate named `first_durable_function` with the dependencies
above and put this in `src/main.rs`. The handler runs a step, suspends on a
two second timer, then runs a second step that reads the first result.

```rust no_run
use std::time::Duration;

use aws_durable_execution_sdk as durable;

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
    durable::init_default_subscriber()?;
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

By default, `aws lambda invoke` invokes synchronously. Lambda keeps the
invocation open until the durable execution completes, or until the shortest
of the function timeout, execution timeout, and 15 minutes. A client can time
out sooner.

```sh
aws lambda invoke --function-name first-durable-function \
    --qualifier '$LATEST' --payload '{}' \
    --cli-binary-format raw-in-base64-out response.json
```

For this execution, the CLI writes the execution result, `"hello, world"`, to
`response.json` and prints invoke metadata that includes `StatusCode` 200 and
`DurableExecutionArn`.

Invoke asynchronously with `--invocation-type Event` when the client should
return without waiting for the execution to finish:

```sh
aws lambda invoke --function-name first-durable-function \
    --qualifier '$LATEST' --invocation-type Event \
    --payload '{}' \
    --cli-binary-format raw-in-base64-out response.json
```

The `Event` invocation leaves `response.json` empty and prints invoke metadata
that includes `StatusCode` 202 and `DurableExecutionArn`. Take that ARN from
the metadata and poll the execution until it leaves `RUNNING`. It reaches
`SUCCEEDED` a couple of seconds after the timer fires, with `"hello, world"` as
its result.

```sh
aws lambda get-durable-execution \
    --durable-execution-arn <execution-arn> \
    --query '[Status,Result]' --output text
```

## The handler and the context

A durable handler is an async function of two arguments that returns a
`Result`.

```rust ignore
async fn handler(
    event: MyEvent,
    ctx: durable::DurableContext,
) -> Result<MyOutput, durable::BoxError>
```

`durable::run` registers it with the Lambda runtime and drives one execution
per invocation. The event type needs `Deserialize` and the output type needs
`Serialize`; both are yours to choose. `BoxError` is
`Box<dyn Error + Send + Sync>`, so `?` accepts any error type without a
conversion.
Use `durable::run_with_options(handler, options)` to start the Lambda runtime
with the supplied `Options`. Use `durable::wrap(handler, options)` to obtain
the service function without starting the runtime, for example when you need
to register it with `lambda_runtime` behind middleware.

Checkpoint writes from concurrent operations batch into fewer requests by
default, and `Options::builder().checkpoint_batching(false)` sends each one
in its own request.

A handler failure does not fail the Lambda invocation. The SDK reports it
inside a successful invocation response, as a `FAILED` envelope the durable
execution service reads, which is what moves the execution to the `FAILED`
status. Lambda-level error signals therefore stay quiet: the CloudWatch
`Errors` metric does not fire, dead-letter queues and `OnFailure`
destinations do not trigger, and X-Ray does not mark the trace as an error.
Monitor and alarm on the durable execution status instead, via
`get-durable-execution` or `list-durable-executions-by-function`.
[Differences from the JavaScript SDK](docs/differences-from-js.md#errors)
lists where the recorded error types differ from the JavaScript SDK's.

A serialized result over 6 MB minus 50 bytes (6 291 406 bytes) is
checkpointed as the execution result, and the envelope carries an empty
`Result`, as in JS.

Durable operations are normal Rust futures. A waiting operation stays
pending. While other SDK work runs, the SDK refreshes service state and may
continue the operation in the same invocation. When no SDK work remains,
the invocation ends with `PENDING`, every future is dropped, and replay
resumes them later. A durable wait is a possible invocation boundary, not a
certain one, so correctness must not depend on which. Rust returns
`PENDING` only when no SDK-owned local work can make progress and at least
one attached service wait remains. Tokio tasks, blocking tasks and threads
that your code creates are outside that guarantee.

An `Err` is a recoverable operation failure. A panic fails the Lambda
invocation. A panic in a step body, a map item, a parallel branch or a
combinator input propagates unchanged, the SDK records nothing for it, and
the service invokes the function again. A panic that happens on every
attempt therefore repeats until the execution times out. Return an `Err` to
fail an operation. A panic in a task whose handle was dropped also fails
the invocation. When several panics happen in one invocation, the SDK
resumes the first one it records and drops the others. Before the panic
leaves the SDK, the SDK cancels every task it started and waits until each
has stopped. With `panic = "abort"`
the process exits at the panic and no cleanup runs.

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

```rust ignore
let wait = ctx.wait(Duration::from_secs(5)).name("timer").spawn();
let work = ctx.step(|_| async { Ok(42_i32) }).name("compute").spawn();

let (timer, result) = tokio::join!(wait, work);
timer?;
let value = result?;
```

`.spawn()` of a step, a wait, a child context, a `map`, a `parallel`, a
`parallel_mixed`, a `create_callback`, a `wait_for_callback` and a
`with_retry`, and `.spawn_batch()` of a `map` and a `parallel`, start the
operation inside the call, as JS does. Before the call returns, the SDK runs
the task-ownership check on the calling task and, when the operation's state
requires a `START`, admits that `START` to the write order. So the `START`
reaches the service before every update the caller sends afterwards. A
`.spawn()` from a task or thread that does not own the context fails the
execution with a context usage error, returns a handle that resolves to the
task-ownership error, queues no `START` and runs no body. A plugin start hook
that JS fires when it queues the `START`, and the replay start hook of a
record that resumes work, also fire before the call returns;
`on_attempt_start` and the body run on the spawned task. `.spawn()` of an `invoke`, a
`wait_for_condition` and a combinator starts the whole operation on the
spawned task instead.

A child context, a `map`, `parallel` or `parallel_mixed` container, each
map item and `parallel` or `parallel_mixed` branch with
`NestingMode::Normal`, the outer context of `wait_for_callback`, and the
outer context of a `with_retry` block with its own record queue their
`START`, fire `on_operation_start`, and start their body without waiting for
the `START` response, as JS does. Their outcome write waits for that
response, and when the `START` write fails, the invocation ends without an
outcome for the operation. An item or branch with `NestingMode::Flat`, and a
`with_retry` block with `.without_child_context()`, write no record of their
own, so they queue no `START` of their own; the attempts of the block queue
theirs.

Dropping the handle that `.spawn()` returns loses only the result. The
operation keeps running until it settles, its scope closes or the
invocation ends. An operation that settles records its outcome and fires
its plugin hooks. Its error goes into its record and is returned to
nobody, and a panic in it fails the invocation. A running detached
operation keeps the invocation from suspending; one that waits only on the
service, such as a wait, lets it suspend. When the handler returns, the
invocation ends and stops each detached operation that still runs. A
stopped operation fires no end hook. Its record, if it has one, keeps the
last state written, which can be `STARTED`, and the execution can end with
the handler's result. The same holds for `.spawn_batch()` and
`Callback::spawn_result()`.

The SDK stops work in three cases. A combinator stops the `.future()`
inputs it started that still run when it resolves, and lets an input
started with `.spawn()` run. A closed scope stops the work still running
inside it: a scope closes when a child context's body returns, when a
`with_retry` block with its own record ends, when an attempt of any
`with_retry` block ends, and when a batch item or the
whole batch completes. The end of the invocation stops all remaining
operation work. A stopped task stops at its next `.await`, and its record
can stay `STARTED`. The SDK cannot stop code between two `.await`s,
`spawn_blocking` work, tasks or threads the handler starts itself, or a
request already sent to another service.

### What determinism requires

The SDK claims each operation's ID synchronously at the call site, so replay
depends on your handler creating the same operations in the same order every
time. Keep the control flow that decides which operations to create a pure
function of the event and of results the SDK already recorded. Iterating a
`HashMap` or `HashSet` to create operations breaks that, because the iteration
order changes between runs; sort into a `Vec` first.
[Differences from the JavaScript SDK](docs/differences-from-js.md#operations)
lists where operation IDs, operation starts and suspension differ from the
JavaScript SDK.

Combinators and `with_retry` blocks are the exceptions to the call-site
rule. A combinator (`try_join_all`, `join_all`, `select_ok` or `race`)
claims its ID late because the context method only collects the
inputs, so inputs that an iterator creates take their IDs first. The
combinator claims its own ID when it is first awaited or polled, or when it
is spawned, as a JS combinator claims its ID when it is first awaited. That
point is in the handler's own control flow, so it is deterministic as long
as the handler awaits the combinator where it awaited it on the first run.
A combinator that is dropped before that point claims no ID and records
nothing. A combinator passed to another combinator must already have
claimed its ID: start the inner combinator with `.spawn()` and pass the
returned handle. A still-unclaimed one fails the execution with a context
usage error.

A `with_retry` builder claims no ID at the call, because its mode is known
only when the builder becomes its future. A block with its own record
claims its ID at `.await`, `.future()` or `.spawn()`. A block with
`.without_child_context()` claims no ID of its own: its first attempt
claims the caller's next ID when it starts, at its first poll, at
`.spawn()` or when a combinator starts it, and each later attempt and wait
claims the caller's next ID when it starts. These points are in the
handler's own control flow too, so they are deterministic as long as the
handler converts and awaits the block where it did on the first run.

```rust ignore
let a = ctx.step(|_| async { Ok(1u32) }).name("a").future();
let b = ctx.step(|_| async { Ok(2u32) }).name("b").future();
let inner = ctx.race([a, b]).name("inner").spawn(); // claims its ID here
let c = ctx.step(|_| async { Ok(3u32) }).name("c").future();
let all = ctx.try_join_all([inner, c]).name("all"); // claims nothing yet
let values = all.await?; // claims its ID here
```

A `map`, `parallel` or `parallel_mixed` takes one ID from its caller, its
own. Its items are numbered inside the batch: item `i` of the batch has the
ID of item `i` in the batch's own namespace, whatever the item count, the
concurrency or the order in which items finish. So an operation the handler
starts beside a batch, for example with `.spawn()` and `tokio::join!`, gets
the same ID on every run.

Put nondeterminism inside a step body instead of between operations. A step
body may read the clock, generate a random ID, or call a service, because only
its recorded result takes part in replay. `StepContext`, the argument a step
body receives, exposes just `attempt()` and no operation methods, so calling a
durable operation inside a step body fails to compile rather than failing at
run time. A `DurableContext` clone that a step body, a `wait_for_condition`
check, or a child body uses for an operation it does not own fails the
execution on the first invocation with a context usage error.

Use `ctx.race` or `ctx.select_ok` rather than `tokio::select!` over durable
futures, so the SDK records which branch won. `tokio::join!` over SDK
operations is allowed: the invocation suspends only when no SDK work can make
progress. Work that is not an SDK operation belongs in a step; awaiting it
between operations lets the invocation suspend. `tokio::task::yield_now()`
outside a step body is such an await too: while an operation waits and no
other SDK work runs, the invocation suspends at it. When an execution
suspends, the SDK drops your handler future, so never make correctness
between operations depend on `Drop` order. An operation claimed, spawned or
awaited on a task or thread that does not own the context, plugin hook work
included, fails the execution with a context usage error.

### Logging

Handler code between operations runs again on every replay, so a `tracing`
record it emits would be written again on every invocation. Install the
subscriber with `durable::init_default_subscriber()`, as the quick start
does, and each record between operations is written once per execution.
The helper has the settings of `lambda_runtime`'s default subscriber: it
reads the level from `AWS_LAMBDA_LOG_LEVEL`, else `RUST_LOG`, else `INFO`,
and writes JSON when `AWS_LAMBDA_LOG_FORMAT` is `json`. It returns an error
when a global subscriber is already installed.
[Differences from the JavaScript SDK](docs/differences-from-js.md#logging)
compares this with the JavaScript SDK's replay-aware logger.
`durable::init_default_subscriber_with_writer(writer)` does the same with
another writer, such as a non-blocking one.

Each record carries correlation fields from the SDK's spans. Every record
has `executionArn`, `requestId`, `isReplay` and, when the invocation has
one, `tenantId`. A record inside a child context body, a map item or a
parallel branch also has the child operation's `operationId` and, when the
operation has a recorded name, `operationName`. A map item or parallel
branch without a caller-given name has its generated name, such as
`map-item-0`. A record inside a step body or a
`wait_for_condition` check has those two fields for the step or the
condition operation, plus `attempt`. The `observability` module lists
which span carries which field.

A custom subscriber keeps the suppression by adding `ReplayFilterLayer` as
the filter of its `fmt` layer. The filter implements the `Filter` trait of
`tracing-subscriber`, which `lambda_runtime` re-exports as
`lambda_runtime::tracing::subscriber`, so no extra dependency is needed.
The `ReplayFilterLayer` rustdoc shows how to keep the records of one target
on every invocation.

```rust ignore
use durable::lambda_runtime::tracing::subscriber;
use subscriber::Layer as _;
use subscriber::layer::SubscriberExt as _;
use subscriber::util::SubscriberInitExt as _;

subscriber::registry()
    .with(subscriber::fmt::layer().json().with_filter(durable::ReplayFilterLayer::new()))
    .init();
```

The SDK never installs a subscriber implicitly. `run`, `wrap` and
`LocalRunner` install none; only a call to `init_default_subscriber` or
`init_default_subscriber_with_writer` does.

### Replacing the transport

The SDK makes two durable execution service calls: it checkpoints
operation updates, and it reads the execution state one page at a time.
By default it makes them through `LambdaExecutionClient`, which wraps an
`aws-sdk-lambda` client. Implement the `ExecutionClient` trait to send
them somewhere else, for example to another compute platform or to a test
fake, and pass it to `OptionsBuilder::execution_client`. The trait speaks
the service model types that `durable::lambda_types` re-exports.

A transport returns an `ExecutionClientError` with one of two scopes.
`ExecutionClientErrorScope::Invocation` fails the invocation, so the
service invokes the function again and replay resumes.
`ExecutionClientErrorScope::Execution` fails the execution. Transient
retry is the transport's job: the SDK treats every error as final for the
invocation.

`execution_client` replaces the Lambda transport, so `build()` rejects it
together with `lambda_client` or `sdk_config`. To decorate the default
transport, for example to add metrics, build it with
`LambdaExecutionClient::new` and wrap it in your own `ExecutionClient`.

```rust ignore
let client = durable::LambdaClient::new(&sdk_config);
let transport = MeteredTransport::new(durable::LambdaExecutionClient::new(client));

let options = durable::Options::builder()
    .execution_client(std::sync::Arc::new(transport))
    .build()?;
durable::run_with_options(handler, options).await
```

## Operations

| Operation | Behavior |
| --- | --- |
| `ctx.step(f)` | Runs `f` and checkpoints its result. On replay, returns the recorded result without re-running `f`. |
| `ctx.wait(duration)` | Waits for `duration`. The invocation suspends during the wait only when no other SDK work can make progress. |
| `ctx.invoke(function_id, input)` | Calls another durable function and waits for its result. |
| `ctx.run_in_child_context(f)` | Runs `f` against a child context with its own operation namespace. |
| `ctx.with_retry(f)` | Runs `f` against a child context and retries the whole block as a unit, with a fresh operation namespace per attempt. |
| `ctx.wait_for_condition(check, state)` | Polls `check`, carrying state between attempts, until the wait strategy completes. |
| `ctx.create_callback()` | Mints a callback ID now and resolves when an external system completes it. |
| `ctx.wait_for_callback(submitter)` | Mints the ID, hands it to `submitter`, and waits for the payload. |
| `ctx.map(items, f)` | Runs `f` per item, each in its own child context. |
| `ctx.parallel(branches)` | Runs named branches concurrently. |
| `ctx.parallel_mixed(branches)` | Runs named branches that each return their own type; each branch handle returns its result after the parallel completes. |
| `ctx.try_join_all(futures)` | Collects every result, failing on the first error. |
| `ctx.join_all(futures)` | Collects every outcome as `Result<O, OperationError>`, never failing fast. |
| `ctx.select_ok(futures)` | Returns the first success and cancels the `.future()` inputs still running. |
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

A step body is `FnOnce + Clone`. The SDK keeps the original closure and calls
a fresh clone of it for each attempt. Each capture follows its type's `Clone`
implementation: a cloned `Vec` or `String` has its own contents, and a cloned
`Arc` shares its allocation. A later invocation runs the handler again and
builds the step closure again; state that comes from outside that handler call
can still be shared. A retry whose delay ends while other work runs makes its
attempt in the same invocation. Wrap a capture that is not `Clone` in an
`Arc`. A value the body must consume, such as a `oneshot::Sender`, can go in an
`Arc<std::sync::Mutex<Option<_>>>`; every clone shares that `Arc`, so the first
attempt that takes the value consumes it.

```rust ignore
let value = ctx
    .step(|_| async { Ok("step completed".to_owned()) })
    .await?;
```

Add `.retry_strategy(...)` to turn transient failures into retries. The
strategy takes the error and the 1-based attempt number and returns
`RetryDecision::Retry { delay }` or `RetryDecision::Stop`. Between attempts
the step waits for the delay. The invocation suspends during that wait only
when no other SDK work can make progress, so a lone retrying step does not
hold the invocation open.

```rust ignore
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

A step without a strategy retries with `RetryStrategyConfig::default_preset()`,
the JS `default` preset: 6 total attempts, a 5 second initial delay doubling
up to 60 seconds, and full jitter. `.retry_strategy_config(...)` takes a
`RetryStrategyConfig` instead of a closure. `RetryStrategyConfig::builder()`
fills the fields it leaves unset as JS `createRetryStrategy` does, with 3
attempts and a 300 second cap, and its `retry_if(predicate)` retries only the
errors for which the predicate returns `true`.

`StepSemantics::AtLeastOncePerRetry` is the default: the SDK re-executes the
body on resume if the previous attempt started but recorded no outcome.
The body starts without waiting for the `START` checkpoint. The outcome is
written only after `START` is acknowledged; if `START` fails, the outcome is
not written and the invocation fails.
`AtMostOncePerRetry` treats that interruption as a failure and consults the
retry strategy, so a non-idempotent body never runs twice for the same
attempt. The strategy sees `StepErrorKind::Interrupted`, and the recorded
failure carries the error type `StepInterruptedError`.

A step records the error its body returns. An unwrapped error records the
type `Error`, because Rust erases the concrete type when the error is
boxed. Wrap the error in `TypedError` to record a type of your choice, and
add `with_error_data` to record an opaque payload as the error data. When
the retry strategy stops, the step returns `StepErrorKind::ExecutionFailed`,
and its `source()` is the body's error. After a replay,
`OperationError::wire()` returns the recorded type and data.

```rust ignore
let result = ctx
    .step(|_| async {
        let err = charge_card().await.unwrap_err();
        Err::<(), durable::BoxError>(Box::new(
            durable::TypedError::with_type("PaymentError", err)
                .with_error_data(r#"{"code":42}"#),
        ))
    })
    .name("charge")
    .retry_strategy(|_err, _attempt| RetryDecision::Stop)
    .await;

if let Err(err) = result {
    let wire = err.wire();
    // Some("PaymentError") and Some(r#"{"code":42}"#), live and after a replay.
    let error_type = wire.and_then(durable::WireError::error_type);
    let error_data = wire.and_then(durable::WireError::error_data);
}
```

A step records the subtype `Step`, and `.sub_type(...)` records an
`OperationSubType` of your choice instead, so history and plugins can tell
the steps of a composed operation apart. `OperationSubType::new` accepts 1
to 32 characters from `A-Z`, `a-z`, `0-9`, `-` and `_`, and rejects an
empty string. Replay compares the subtype, so changing it on an existing
step fails the executions in flight with `NonDeterministicExecutionError`.

```rust ignore
let receipt = ctx
    .step(|_| async { Ok(charge_card().await?) })
    .name("charge")
    .sub_type(durable::OperationSubType::new("Payment")?)
    .await?;
```

### wait

```rust ignore
ctx.wait(Duration::from_secs(60)).name("cooldown").await?;
```

The duration is rounded up to whole seconds, so a wait never ends early, and
a wait shorter than one second waits one second. A wait that ends while
other work runs continues in the same invocation.

### invoke

`invoke` calls another durable function and resolves with its result. The
output type parameter comes first so you can turbofish it.

```rust ignore
let receipt = ctx
    .invoke::<serde_json::Value, _>(&target_function_name, event)
    .name("delegate")
    .await?;
```

An invoke whose target finishes while other work runs continues in the same
invocation.

### wait_for_condition

`wait_for_condition` runs a check repeatedly, carrying checkpointed state
between attempts, and waits between polls. The invocation suspends during
that wait only when no other SDK work can make progress, and a check that
becomes due while other work runs continues in the same invocation.
`wait_strategy_fn` receives the
current state and the attempt number and returns `WaitDecision::complete()`
or `WaitDecision::continue_with(delay)`.

```rust ignore
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

Pass a `WaitStrategy` to `.wait_strategy(...)` instead when a predicate
plus attempt cap expresses the condition. A `WaitStrategy` carries a
required completion predicate over the state, a `max_attempts` cap, and
exponential backoff configuration (`initial_delay`, `max_delay`,
`backoff_rate`, `jitter`). After each check it completes the operation
when the predicate matches the new state, fails it with a
`MaxChecksExceeded` error once `max_attempts` checks have run, and
otherwise waits for the computed backoff delay. Build one with
`WaitStrategy::builder(predicate)` (the predicate is the builder's
argument, so an unbounded strategy cannot be constructed), and the
remaining knobs default to 60 attempts, 5 second initial delay, 5 minute
maximum delay, 1.5 backoff rate, and full jitter, matching the JS and
Python SDKs. With no strategy set at all, the check runs exactly once and
the operation completes with that state. A check that becomes due while
other work runs continues in the same invocation.

### callbacks

`create_callback` mints an ID and returns a `Callback` immediately, so you can
hand the ID to an external system and await the payload separately. An
external caller completes it with the
`SendDurableExecutionCallbackSuccess` API.

```rust ignore
let cb = ctx.create_callback::<String>().name("approval").await?;
tracing::info!(callback_id = %cb.id(), "awaiting external completion");
let approval = cb.result().await?;
```

`wait_for_callback` collapses those two halves into one operation: it mints
the ID, passes it to your submitter closure, and resolves with the payload.
The submitter receives the ID as an owned `String`, so move it straight into
the async block.

```rust ignore
let approval: String = ctx
    .wait_for_callback::<String, _, _>(|_step_ctx, callback_id| async move {
        publish_approval_request(callback_id).await?;
        Ok(())
    })
    .name("await-approval")
    .await?;
```

Both builders decode the payload with `JsonSerdes` by default.
`.serdes(PassthroughSerdes)` returns the raw string the external caller
sent, as JS does by default:

```rust ignore
let cb = ctx.create_callback::<String>().serdes(PassthroughSerdes).await?;
let raw: String = cb.result().await?;
```

`wait_for_callback` records the raw payload as its result and applies the
serdes after it, once per invocation that reads the result.

Both builders take `.timeout(...)` and `.heartbeat_timeout(...)`.
`wait_for_callback` also takes `.submitter_retry(...)`, which retries the
submitter with the same retry strategy shape a step uses.
A submitter that fails after its retry strategy stops surfaces as
`CallbackErrorKind::SubmitterFailed`. `wait_for_callback` reports an
external failure with the type `Callback.ExternalFailure` and keeps the
external message and error data, while `create_callback` keeps the
external caller's type.
`create_callback` also takes `.sub_type(...)`, which records an
`OperationSubType` instead of `Callback`, as on a step.

A callback that an external system completes while other work runs
continues in the same invocation. `cb.spawn_result()` waits for the result
on its own task, as `.spawn()` does for other operations. Dropping its
handle loses only the result. The wait keeps running until the callback
settles, its scope closes or the invocation ends. A wait that settles
records its outcome and fires its plugin hooks. Its error is returned to
nobody, and a panic in it, such as one in its serdes, fails the
invocation. While the wait decodes the payload it keeps the invocation from
suspending, and while it waits for the callback it lets the invocation
suspend, because it waits only on the service. When the handler returns,
the invocation ends and stops the wait if it still runs. The stopped wait
fires no end hook, the callback's record keeps the last state written,
which can be `STARTED`, and the execution can end with the handler's
result.

### child contexts

A child context gets its own operation namespace, which keeps a subroutine's
operation IDs independent of the caller's.

```rust ignore
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

When a child context's body returns, an operation it started and did not
await writes no further checkpoint, and awaiting it never resolves.

The body should run at least one durable operation. Its result is saved,
so a replay returns it without running the body, with three exceptions: a
body that returns an error is not retried as a step is, a result over
256 KiB makes the body run again on every replay to rebuild it, and so does
a context that `plugin_child_operations_depth` preserves. So put plain
computation in a step. An oversized result records the payload `""`, or the
output of `.summary(...)`, which receives the result and is called only
when the serialized result is over 256 KiB. The summary is for
observability: replay never reads it, and it must stay under 256 KiB.
`with_retry` takes the same `.summary(...)` for its block.

A child context records the subtype `RunInChildContext`, and
`.sub_type(...)` records an `OperationSubType` of your choice instead, as
on a step. Replay compares the subtype, so changing it on an existing
child context fails the executions in flight with
`NonDeterministicExecutionError`.

```rust ignore
let order = ctx
    .run_in_child_context(|child| async move {
        Ok(child.step(|_| async { Ok(reserve().await?) }).name("reserve").await?)
    })
    .name("checkout")
    .sub_type(durable::OperationSubType::new("Checkout")?)
    .await?;
```

`.virtual_context()` runs the body without recording the child context
itself, as JS `virtualContext` does. The body still gets its own operation
namespace, but no record is written for the child: its operations name the
caller's parent as their parent, and the child adds no level to
`plugin_child_operations_depth`. Because nothing records its result, the
body runs inline on the awaiting task in every invocation, and its
operations replay their recorded results. A body `Err` returns a
`ChildContextError` of kind `ChildFailed` wrapping the body's error, and a
serdes failure returns the serdes error, or ends the invocation when it is
marked `RetryableSerdesError`. The child stops with its caller's scope, so
an operation its body started and did not await keeps checkpointing after
the body returns. A record at the child's ID fails replay with
`NonDeterministicExecutionError`, so switching an existing child to virtual
fails the executions in flight. A summary set on a virtual child is never
called, because no record exists to hold it, and its subtype appears only
in errors and plugin hook infos. Its user-function hooks and
`wrap_user_function` run in every invocation, because the body runs in
every invocation.

### with_retry

`with_retry` runs a closure against a child context and applies a retry
strategy to the closure's overall outcome, so a multi-operation block
retries as a unit. Each attempt gets a fresh child operation namespace: a
failed attempt's recorded operations are never replayed into the next
attempt, so every operation in the block re-runs on retry. The block waits
between attempts, exactly as a retrying step does: the invocation suspends
during that wait only when no other SDK work can make progress, and a wait
that ends while other work runs continues in the same invocation. The retry
progress is derived from checkpointed results, so it survives suspension.
The closure is `Fn` rather than `FnOnce` because the SDK calls it once per
attempt.
The closure's second argument is the 1-based attempt number, `N` in the
attempt's name `attempt-N`, as metadata for names and logs: a resumed
attempt receives the same number, and operation IDs do not depend on it.

```rust ignore
let receipt = ctx
    .with_retry(|child, attempt| async move {
        tracing::info!(attempt, "quoting and booking");
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
closure, and without either the step default applies,
`RetryStrategyConfig::default_preset()` (6 total attempts with exponential
backoff). `RetryStrategyConfig::linear()` retries
with delays of 1 to 5 seconds and `RetryStrategyConfig::no_retry()` makes
the first failure final, as the JS `linear` and `noRetry` presets do.

The wait after failed attempt `N` is named `{name}-backoff-N` when the
block has a non-empty name, as in the JS SDK. The waits of an unnamed block
are unnamed.

`.sub_type(...)` on `with_retry` labels the block's own child context; each
`attempt-N` child context keeps `RunInChildContext`, and each wait between
attempts keeps `Wait`.

`.without_child_context()` runs the attempts in the caller's context
without a record for the block, as JS `wrapWithRunInChildContext: false`:
the block claims no operation ID, a block named `b` names its attempts
`b-attempt-N` and its waits `b-backoff-N`, `sub_type` has no recorded
effect, and no plugin hook runs for the block itself.

`.virtual_context()` makes the block a virtual child context, as on
`run_in_child_context`: the block claims an ID and runs its attempts in its
own namespace, but writes no record, so its attempts and waits name the
caller's parent as their parent. A block named `b` names its attempts
`b-attempt-N` and its waits `b-backoff-N`; an unnamed block names its
attempts `attempt-N` and leaves its waits unnamed. The block's summary is
never called, its subtype appears only in errors and plugin hook infos,
and its operation hooks report `is_virtual` true. The later call of
`.virtual_context()` and `.without_child_context()` wins.

A `with_retry` builder claims no ID at the call. A block with its own
record, and a virtual block, claim their ID when the builder becomes its
future, at `.await`, `.future()` or `.spawn()`.

When retries exhaust, the operation fails with a `ChildContextError` of
kind `ChildFailed` whose source is the last attempt's error, also for a
virtual block. With `.without_child_context()`, it returns the last
`attempt-N` child's error unchanged.

### map and parallel

`map` applies one closure to every item, giving each item its own child
context. Items need `Serialize + DeserializeOwned` because the SDK records
them.

```rust ignore
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

```rust ignore
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

History records a name for each item's child context. `.item_namer(...)`
names a map item: the namer receives the item and its index, and runs once
per item when the map starts, so it must be deterministic. A map item
without a namer, or whose namer returns an empty string, records
`map-item-{index}`. A branch built with `Branch::unnamed(...)`, or with an
empty name, records `parallel-branch-{index}`. A completion predicate sees
the name the caller gave instead: `None` for a map item without a namer and
for an unnamed branch, and `Some("")` for an empty name.

Both accept `.max_concurrency(...)` and `.completion(...)`. By default
(with no `.completion(...)` call, or with an empty `CompletionConfig`) the
first item failure fails the batch, matching the Python and JS SDKs.
Configuring any criterion, including a predicate, replaces that implicit
fail-fast: use `with_tolerated_failure_count(0)` for explicit fail-fast, or
`with_tolerated_failure_percentage(100.0)` to tolerate all failures.

When a batch result is over 256 KiB, the batch records a summary of its
outcome instead, and `.summary(...)` adds the output of a function of the
`BatchResult` under that record's last key, `summary`, which replay never
reads.

A `CompletionConfig` can end the batch early with fixed thresholds.
`with_min_successful(n)` stops once `n` items succeed, and
`with_tolerated_failure_count(n)` and `with_tolerated_failure_percentage(pct)`
fail the batch once the failures exceed the tolerance.
`CompletionConfig::builder()` combines thresholds, and the first to fire
wins. After each settle the SDK checks them in JS's order: the failure
tolerance first, then "every item finished" (`ALL_COMPLETED`), then the
minimum. So `min_successful(2)` over two items records `ALL_COMPLETED`.
`min_successful(0)` completes a non-empty batch at its first settle. The
percentage is an `f64` and accepts a fraction such as `12.5`; the batch fails
when the failure rate strictly exceeds it. The SDK compares
`failures * 100 > pct * total` rather than dividing, so 7 failures of 50 do
not exceed `14.0`. `build()` rejects a percentage that is not finite or lies
outside `0.0..=100.0`.

The builder's `.completion_predicate(...)` sets a custom rule instead. The
predicate receives a `BatchStats` snapshot and returns a `CompletionDecision`:
`Continue`, `Complete(CompletionOutcome::Succeeded)`, which records
`CUSTOM_COMPLETION_SUCCEEDED`, or `Complete(CompletionOutcome::Failed)`, which
records `CUSTOM_COMPLETION_FAILED`. A batch the predicate completed as failed
has the status `Failed`, and awaiting it returns the first failed item's
error, or a `BatchCompletionFailed` error when no item failed. A predicate
that never completes the batch leaves the reason `ALL_COMPLETED`. As in JS, a
predicate cannot be combined with a threshold: `build()` rejects the mix.

The SDK asks the predicate before each item starts and after each item
settles, and a zero-item batch asks it once. The snapshot counts follow the
settle order, and `items()` holds every input item by index, with the name
the caller gave it and its status: `None` before it starts, `Started` while
it runs, then `Succeeded` or `Failed`. So a predicate can complete the batch
as soon as the items it names have settled, without waiting for earlier
items, and it can complete the batch before the first item starts.

The live settle order is not recorded, and on resume the recorded items
settle first, in index order. So the sequence of snapshots can differ between
runs, and the predicate must be a pure function of `BatchStats` and
monotonic. A predicate is monotonic when, after it returns `Complete(outcome)`
for a snapshot, it returns `Complete(outcome)` with the same outcome for every
later snapshot of the same batch, where each item's status is the same or
further along. The SDK stops asking once the first trigger fires, so within
one run a predicate is never asked to reverse a decision; monotonicity is
what keeps two runs in agreement. A predicate such as
`succeeded() == 1 && failed() == 0` is not monotonic: it can complete on one
run and never complete on resume, as
[JS's own advice](https://github.com/aws/aws-durable-execution-sdk-js/blob/67549feacd5568ea305b7c2d533d3549974f8c9c/packages/aws-durable-execution-sdk-js/src/types/batch.ts#L276-L282)
explains. A rule such as "item 0 succeeded, or items 1 and 2 both
succeeded" is monotonic.

```rust ignore
let quorum = CompletionConfig::builder()
    .completion_predicate(|stats| {
        let ok = |i: usize| {
            stats.items()[i].status == Some(BatchItemStatus::Succeeded)
        };
        if ok(0) || (ok(1) && ok(2)) {
            CompletionDecision::Complete(CompletionOutcome::Succeeded)
        } else {
            CompletionDecision::Continue
        }
    })
    .build()?;
```

`MapBuilder` and `ParallelBuilder` also expose `.await_batch()`:
await it instead of the builder to get a `BatchResult<O>` that reports each
item's status and why the batch ended. `BatchResult::status()` returns a
`BatchStatus` enum, and `BatchResult::errors()` returns `BatchError` entries
that tie each failure to the index and name of the item that produced it,
along with the error's type identifier and message.
`succeeded()`, `failed()` and `started()` iterate the items with that
status in input order, and `started_count()` counts the `Started` ones.
`into_result()` returns the first failed item's error, or the batch
completion error when the status is `Failed`, and the `BatchResult`
otherwise. Unlike `.await`, it returns an error for a failure within the
configured tolerance too.

`.spawn_batch()` starts the batch on its own task at once and returns a
`DurableFuture<BatchResult<O>>`, as `.spawn()` does for the `Vec<O>` form,
so the handler can run other operations before it awaits the batch.
Dropping the returned handle loses only the result. The batch keeps
running until it completes, its scope closes or the invocation ends. A
batch that completes records its outcome and fires its plugin hooks, its
error goes into its record and is returned to nobody, and a panic in it
fails the invocation. While an item runs, the batch keeps the invocation
from suspending, and while every unfinished item waits only on the
service, such as on a wait, it lets the invocation suspend. When the
handler returns, the invocation ends and stops the batch if it still runs.
The stopped batch fires no end hook, its record, if it has one, keeps the
last state written, which can be `STARTED`, and the execution can end with
the handler's result.

```rust ignore
let batch = ctx
    .map(items, |child, item, _idx| async move {
        Ok(child.step(move |_| async move { Ok(item * 10) }).name("scale").await?)
    })
    .name("scale-all")
    .spawn_batch();
ctx.wait(Duration::from_secs(1)).name("meanwhile").await?;
let batch = batch.await?.into_result()?;
```

`.max_concurrency(0)` fails the execution with an error the handler
cannot catch, and the batch records nothing. Without `.max_concurrency`,
concurrency is unlimited. A failure percentage applies to `parallel` as
it does to `map`.

An item that waits on a durable operation keeps its concurrency slot until it
finishes. When its wait completes while other work runs, it continues in the
same invocation.

When the completion rule ends the batch early, items that are still running
are cancelled at their next `.await` and reported as `Started`. A cancelled
item writes no further checkpoint. Work that any item started and that still
runs, including work of an item that already settled, stops too.

### parallel_mixed

`parallel_mixed` runs branches of different result types in one parallel
operation. Add each branch to a `BranchSet`. `add` returns a
`BranchHandle<T>` typed with that branch's result, and `add_with_serdes`
does the same with a serdes of your choice for that branch. After the
`.await` of `parallel_mixed`, each handle's `into_result()` returns its own
type.

```rust ignore
let mut branches = BranchSet::new();
let reservation = branches.add("inventory", |child| async move {
    Ok(child
        .step(|_| async { Ok(Reservation { sku: "sku-1".to_owned() }) })
        .name("reserve")
        .await?)
});
let auth = branches.add_with_serdes("payment", PaymentSerdes, |_child| async {
    Ok(PaymentAuth { code: 7 })
});

ctx.parallel_mixed(branches).name("process-order").await?;

let reservation: Reservation = reservation.into_result()?;
let auth: PaymentAuth = auth.into_result()?;
```

The operation records the same history as `parallel` over branches with
the same names and outcomes: one parallel parent and one child context per
branch. `ParallelMixedBuilder` takes `.max_concurrency(...)`,
`.completion(...)`, `.nesting(...)`, `.result_serdes(...)`, `.summary(...)`
and `.spawn()`, as `ParallelBuilder` does. `.await` returns the batch's
`CompletionReason`. A handle whose branch settled holds the branch's value
or error, whatever the parallel's own outcome. A handle whose branch never
started, or still ran when the batch completed, returns a `NotCompleted`
error. With `.spawn()`, the handles receive their results when the
returned future is awaited.

The result serdes works as on `ParallelBuilder`: it is a `Serdes<String>`
over the batch record's JSON text, and it defaults to `PassthroughSerdes`.
`JsonSerdes` stores the record as a quoted JSON string.

The summary receives a `BatchResult<BranchValue>`. A `BranchValue` shows
nothing of its value, so the summary reads each value with its branch's
key: take `handle.key()` before the set goes to `parallel_mixed`, and call
`batch.value(&key)`, which returns `Some(&value)` for a branch that
succeeded and `None` otherwise.

```rust ignore
let mut branches = BranchSet::new();
let report = branches.add("report", |_child| async { Ok(build_report().await?) });
let rows = branches.add("rows", |_child| async { Ok(load_rows().await?) });
let (report_key, rows_key) = (report.key(), rows.key());

ctx.parallel_mixed(branches)
    .name("build-report")
    .summary(move |batch| {
        let report_len = batch.value(&report_key).map_or(0, String::len);
        let row_count = batch.value(&rows_key).map_or(0, Vec::len);
        format!("report: {report_len} bytes, rows: {row_count}")
    })
    .await?;
```

Pick the parallel form by the branch result types:

- `parallel`, when every branch returns the same type.
- `parallel` over an enum, when the results are cases of one type that the
  handler processes together. Use serde's default (externally tagged) enum
  representation. `#[serde(untagged)]` can decode a result into the wrong
  variant when two variants accept the same JSON.
- `parallel_mixed`, when each branch returns its own type.

### combinators

The four combinators take `DurableFuture` values, which `.future()` and
`.spawn()` produce, and record the combined outcome as one operation. An input
that waits stays pending. When every remaining input waits and nothing else
runs, the invocation suspends.

When a combinator runs, it begins the start phase of each `.future()` input,
in input order, before its own start step, as a JS combinator awaits inputs
that already run. A step, a wait, a child context, a `with_retry`, a `map`, a
`parallel` or a `wait_for_callback` input queues its `START` then, so its
first checkpoint comes before the combinator's, and a start hook it fires
when it queues `START` runs before the combinator's. A start hook that waits
for the `START` response, that of a wait or of a step under
`AtMostOncePerRetry`, keeps that timing and runs after the combinator's. A
lazy invoke or `wait_for_condition` is polled once, in input order, before
the combinator's start step, and the combinator does not wait for its
`START`. The combinator then queues its own `START`, fires its start hook,
and runs the inputs without waiting for the `START` response; its `SUCCEED`
or `FAIL` waits for it. A replay of a final record starts no `.future()`
input. A handle that `.spawn()` returned already runs independently, and a
combinator never starts it again. Callback creation is not a combinator
input: await `create_callback` first, and pass `Callback::result()`.

A combinator returns the value decoded from the payload it records, so the
first run returns the value a replay returns. A `join_all` item's `Err` is
therefore a child-context error whose `wire()` carries the input's recorded
failure on both paths. When `try_join_all`, `select_ok` or `race` resolves,
it cancels every input it started from `.future()` that is still running,
and a cancelled input stops at its next `.await`. An input started with
`.spawn()` keeps running, and its result is dropped. `join_all` cancels
nothing.

A combined result larger than 256 KiB is not stored in the combinator's
record, which then carries `ReplayChildren`. Replay drives the inputs again
for `try_join_all` and `join_all`. For `select_ok` and `race` the record
stores the winner's input index, and replay drives only that input again and
drops the others without running them. An input whose own terminal record is
inline replays without running again. An input whose own terminal record has
`ReplayChildren` runs its body again to rebuild its value, and its recorded
descendants replay.

`try_join_all` collects every result, propagating the first error:

```rust ignore
let a = ctx.step(|_| async { Ok(1u32) }).name("a").future();
let b = ctx.step(|_| async { Ok(2u32) }).name("b").future();
let c = ctx.step(|_| async { Ok(3u32) }).name("c").future();

let results: Vec<u32> = ctx.try_join_all([a, b, c]).name("gather").await?;
```

`join_all` returns one `Result<O, OperationError>` per input, in input order,
so you can inspect failures individually without failing fast:

```rust ignore
let good = ctx.step(|_| async { Ok(1u32) }).name("good").future();
let bad = ctx
    .step(|_| async { Err::<u32, durable::BoxError>("boom".into()) })
    .name("bad")
    .retry_strategy(|_err, _attempt| durable::RetryDecision::Stop)
    .future();

let outcomes: Vec<Result<u32, durable::OperationError>> =
    ctx.join_all([good, bad]).name("collect").await?;
for outcome in &outcomes {
    match outcome {
        Ok(value) => tracing::info!("ok: {value}"),
        Err(err) => tracing::info!("err: {err}"),
    }
}
```

The two levels of `Result` report different failures. The outer `Err`, which
`?` propagates above, means the combinator itself failed, for example on
replay nondeterminism. A failed checkpoint write never reaches it: the write
fails the invocation, and the combinator does not resolve. An input that returns
`Err` appears only as that input's inner `Err`, and the outer result is then
`Ok`. A panicking input ends the invocation, as a panic does everywhere else
in the SDK: the SDK resumes the original panic payload, and `join_all` writes
no record and returns no result. Nothing cancels an input that `join_all`
awaits, because `join_all` drops no input before it settles.

`select_ok` returns the first success, cancelling the inputs still running:

```rust ignore
let primary = ctx
    .step(|_| async { Err::<u32, durable::BoxError>("unavailable".into()) })
    .name("primary")
    .retry_strategy(|_err, _attempt| durable::RetryDecision::Stop)
    .future();
let fallback = ctx.step(|_| async { Ok(42u32) }).name("fallback").future();

let winner: u32 = ctx.select_ok([primary, fallback]).name("first-ok").await?;
```

`race` returns the first outcome to settle, whether it succeeded or failed:

```rust ignore
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
representation: a custom format sees struct field declaration order,
`i128` values outside the `i64`/`u64` ranges, and everything else the real
type carries.
[Differences from the JavaScript SDK](docs/differences-from-js.md#serialization)
lists where serdes behavior differs from the JavaScript SDK.

The SDK awaits the future a serdes returns directly on the executor
thread. Cheap in-memory transforms (like the default `JsonSerdes`) run
inline in the returned future; an implementation that blocks (filesystem
I/O, long-running synchronous work) must move that work into
`tokio::task::spawn_blocking` itself, the way `FileSystemSerdes` does.
`async fn` in the implementation satisfies the trait's `impl Future`
return type:

```rust ignore
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
format implements `Serdes<ConcreteType>` directly, `impl Serdes<Order>
for OrderSerdes`, and gains compile-time pairing: attaching it to an
operation that produces another type fails to compile.

Serdes are configured **per operation**, with `.serdes(...)` on
`StepBuilder`, `InvokeBuilder`, `ChildBuilder`, `WithRetryBuilder`,
`WaitForConditionBuilder`, `CreateCallbackBuilder`,
`WaitForCallbackBuilder`, `MapBuilder`, or `ParallelBuilder`. A
`parallel_mixed` branch takes its serdes from `BranchSet::add_with_serdes`
instead. Each builder
carries its serdes as a generic type parameter defaulting to `JsonSerdes`,
and `.serdes(...)` swaps that parameter; `.await` still produces a
`DurableFuture<O>` regardless of the serdes type, so futures configured
with different serdes implementations coexist in one combinator input
collection. `WaitBuilder` and the combinator builders carry no serdes
method because their payloads are structural, not user-typed.
`MapBuilder`, `ParallelBuilder` and `ParallelMixedBuilder` also carry a
result serdes, set with `.result_serdes(...)`, which defaults to
`PassthroughSerdes`, not `JsonSerdes`.

On `MapBuilder`, `ParallelBuilder` and `ParallelMixedBuilder` the two
serdes do different jobs.
The item `.serdes(...)` is a `Serdes<O>` for each item's value, as on any
other operation, or for each branch's value on `ParallelMixedBuilder`.
`.result_serdes(...)` is a `Serdes<String>` over the batch record's JSON
text, in the shape of the JS SDK's batch result. It must be lossless: its
deserializer must reconstruct the exact record text its serializer
received, though the stored payload may use any format.
`PassthroughSerdes` stores the record text unchanged, and `JsonSerdes`
stores it as a quoted JSON string.

There is no execution-wide serdes slot: a single trait-object slot cannot
represent `Serdes<T>` for every operation output type without erasing the
value again. To share one instance across a handler, create an `Arc<S>`
and clone it into each operation; `Arc<S>` forwards to `S`:

```rust ignore
let shared = std::sync::Arc::new(Base64JsonSerdes);
let a: String = ctx
    .step(|_| async { Ok("a".to_owned()) })
    .serdes(std::sync::Arc::clone(&shared))
    .await?;
let b: u32 = ctx.step(|_| async { Ok(7_u32) }).serdes(shared).await?;
```

On a live run, an operation that records a value returns
`deserialize(serialize(value))`, not the value the body produced. This
holds for step results, child context and `with_retry` results, map and
parallel item results, and `wait_for_condition` state. Replay returns
`deserialize` of the stored wire string, so live execution and replay
return the same value. A serdes that changes the value is therefore
supported on those paths, and the handler sees the decoded value. The one
requirement is that `deserialize` returns the same value for the same wire
string; `serialize` may change the value. This does not apply to a batch
`result_serdes`, whose deserializer must reconstruct the exact record text
its serializer received. Operations transfer the owned value to
`serialize`, and ownership transfer is also what keeps
operation outputs at `Send` (not `Sync`): the serdes can move the whole
call into a `'static` blocking task without borrowing across an `.await`.

A serdes error takes one of two paths. By default it is permanent: a
serialize failure after the body ran records a terminal failure, so the
body does not run again, and a deserialize failure of a recorded value
returns an error to the handler. An error marked with
`serdes::RetryableSerdesError` is transient: the SDK ends the Lambda
invocation, records nothing, and the service invokes the function again.
The SDK finds the marker anywhere in the error's `source()` chain, and
`FileSystemSerdes` marks the failures of its filesystem calls.

```rust ignore
use durable::serdes::RetryableSerdesError;

impl<T> Serdes<T> for BlobSerdes
where
    T: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
{
    async fn serialize(
        &self,
        value: T,
        context: SerdesContext,
    ) -> Result<String, durable::BoxError> {
        // A value that cannot be encoded is a permanent failure.
        let bytes = serde_json::to_vec(&value)?;
        // A failed upload is transient: mark it, so the SDK ends the
        // invocation and the service invokes the function again.
        let key = self
            .store
            .put(context.operation_id(), bytes)
            .await
            .map_err(RetryableSerdesError::new)?;
        Ok(key)
    }
    // deserialize: read the blob back, marking a failed read the same way.
}
```

`FileSystemSerdes` stores payloads on a durable shared filesystem (Amazon EFS
or S3 Files mounted to Lambda). Every write goes to a new, unique file
(atomically renamed into place) under a directory resolved from the
`SerdesContext`, and the checkpoint envelope records which file to read on
replay, so a retried attempt can never overwrite a file an accepted
checkpoint references. File references are only honored from records the SDK
itself wrote, resolved strictly under the configured base path; externally
delivered payloads (callback completions) are always read as plain values.
Each `serialize` or
`deserialize` call runs its complete implementation (JSON rendering or
parsing, path resolution, and the file I/O) inside one
`tokio::task::spawn_blocking` task, so the executor thread never touches
the filesystem. Do not use it with Lambda's ephemeral `/tmp`: that storage
is local to a single execution environment and does not persist across
invocations. Mount EFS or S3 Files and point `FileSystemSerdes` at the
mount path.

```rust ignore
use durable::serdes::{FileSystemSerdes, FileSystemSerdesConfig, FileSystemSerdesMode};

let serdes = FileSystemSerdes::with_config(
    "/mnt/efs",
    FileSystemSerdesConfig::builder()
        .storage_mode(FileSystemSerdesMode::Overflow)
        .build(),
);
```

A file envelope holds only the file's path, so the checkpoint shows nothing
of the value. `FileSystemSerdesConfigBuilder::preview` sets a function that
receives the value's JSON text and returns a JSON object as text, which the
envelope carries under a `preview` key beside the path. The function runs
only when the value goes to a file, and reading ignores the preview.
`build_preview` builds one by field selection and masking, as the JS SDK's
`buildPreview` does: `PreviewMode::ExcludeAll` starts with no field shown,
`include` shows a field, `mask` shows it with `"***"`, and the preview stops
at `max_preview_bytes` (default 4096). A field given as a string matches
that name at any depth, and `PreviewField::path` matches one dotted path.

```rust ignore
use durable::serdes::{
    FileSystemSerdes, FileSystemSerdesConfig, PreviewConfig, PreviewMode, build_preview,
};

let preview = PreviewConfig::builder(PreviewMode::ExcludeAll)
    .include("orderId")
    .mask("email")
    .build();
let serdes = FileSystemSerdes::with_config(
    "/mnt/efs",
    FileSystemSerdesConfig::builder()
        .preview(move |json| Ok(build_preview(json, &preview)))
        .build(),
);
// The envelope of {"orderId":"o-1","email":"a@example.com","items":[...]}:
// {"__aws_durable_serdes":{...},"file":"...","preview":{"orderId":"o-1","email":"***"}}
```

## Instrumentation plugins

A plugin observes an execution for tracing and telemetry without changing
it. `Options::builder().plugin(..)` takes a `PluginFactory`, which the SDK
calls once per invocation and which holds the state shared across
invocations. The plugin the factory returns serves only that invocation.
The SDK releases it when the invocation returns; a `DurableContext` clone
that outlives the invocation keeps it alive until that clone drops. Every
hook of the `Plugin` trait defaults to a no-op, so a plugin implements only
the hooks it needs. The `plugin` module's rustdoc gives each hook's full
contract, and [Differences from the JavaScript SDK](docs/differences-from-js.md#plugins)
lists where plugins behave differently from the JavaScript SDK.

Example 1 counts the `on_operation_end` calls of each invocation and adds
them to a total that the factory keeps across invocations.

```rust no_run
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use aws_durable_execution_sdk as durable;
use durable::Options;
use durable::plugin::{
    HookFuture, HookWork, InvocationEndInfo, InvocationInfo, OperationInfo, Plugin,
    PluginFactory,
};

/// Keeps the total across invocations. `PluginFactory` has no `Debug`
/// bound, so the factory needs none.
#[derive(Default)]
struct CountingFactory {
    total: Arc<AtomicU64>,
}

impl PluginFactory for CountingFactory {
    fn create_plugin(
        &self,
        _info: &InvocationInfo,
    ) -> Result<Option<Box<dyn Plugin>>, durable::BoxError> {
        Ok(Some(Box::new(CountingPlugin {
            ends: AtomicU64::new(0),
            total: Arc::clone(&self.total),
        })))
    }
}

/// Counts the `on_operation_end` calls of one invocation.
struct CountingPlugin {
    ends: AtomicU64,
    total: Arc<AtomicU64>,
}

impl Plugin for CountingPlugin {
    fn on_operation_end(&self, _info: &OperationInfo) -> HookWork {
        self.ends.fetch_add(1, Ordering::Relaxed);
        Ok(None)
    }

    fn on_invocation_end<'a>(&'a self, info: &'a InvocationEndInfo) -> HookFuture<'a> {
        let ends = self.ends.load(Ordering::Relaxed);
        self.total.fetch_add(ends, Ordering::Relaxed);
        tracing::info!(status = ?info.status(), ends, "invocation ended");
        Box::pin(std::future::ready(Ok(())))
    }
}

async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    let greeting = ctx
        .step(|_| async { Ok("hello".to_owned()) })
        .name("greet")
        .await?;
    Ok(greeting)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::init_default_subscriber()?;
    let options = Options::builder()
        .plugin(CountingFactory::default())
        .build()?;
    durable::run_with_options(handler, options).await
}
```

### Invocation hooks

`on_invocation_start` and `on_invocation_end` are async. The SDK awaits
`on_invocation_start` before the handler runs. It awaits `on_invocation_end`
once, after every checkpoint of the invocation is written, with the status
`Succeeded`, `Failed`, `Pending` or `Retrying`, so that hook is the place to
flush an exporter. The SDK cuts `on_invocation_end` off 50 ms before the
invocation deadline, and waits for it to finish when the context has no
deadline, as under `LocalRunner`. When a panic ends the invocation,
`on_invocation_end` runs with `Retrying` after the SDK has stopped every
task it started, and the panic then continues. With `panic = "abort"` the
process exits at the panic, and no hook runs. All plugins' hooks run
concurrently.

### Operation hooks

`on_operation_start`, `on_operation_end`, `on_attempt_start`,
`on_attempt_end` and `on_operation_change` are synchronous, and no hook's
synchronous part may block. Each runs on the task that reaches its hook
site, and none runs while the SDK holds a checkpoint lock.
`on_attempt_start` and `on_attempt_end` run on the operation's own task.
`on_operation_start` and `on_operation_end` usually run there too, but the
start hook of an operation that `.spawn()` starts can run on the caller's
task before `.spawn()` returns. `on_operation_change` can run on any task:
an operation's task, a task that sent a refresh, or the invocation's own
task, not necessarily the task of the operation that changed. When a hook
is caused by a checkpoint response in this invocation, that checkpoint
call has returned before the hook runs; replay hooks can instead describe
state supplied at invocation bootstrap. A hook that needs async work
returns it as `HookWork`. The SDK runs that work on its own task, outside
the operation, and waits for it before `on_invocation_end` until 500 ms
before the invocation deadline. Work still
running then is aborted, and the SDK waits for each aborted task to end, so
no hook work runs after `on_invocation_end` starts. Work started after that
point is dropped with a WARN.

### Wrap and user-function hooks

`wrap_invocation` brackets each synchronous section of the invocation,
`on_invocation_end` included: every poll of the handler, of each SDK task
that runs an operation, of the end-of-invocation flush and of the drain of
hook work. `wrap_user_function` brackets each poll and each drop of a user
function: a live step attempt or condition check, a child body (including
the outer block of a `with_retry` with its own record, but not the
`attempt-N` contexts of either mode; a block with `.without_child_context()`
has no outer block and so no user function of its own), the body of a
virtual child context or a virtual `with_retry` block, a user function with
`is_virtual()` true whose hooks fire and whose wraps run in every
invocation, the `wait_for_callback` body, each map item, parallel branch and
`parallel_mixed` branch, each map, parallel and `parallel_mixed` container,
and each combinator join. It covers map items, parallel branches and
`parallel_mixed` branches in both nesting modes. Each wrap receives
`next: &mut dyn FnMut()`, and `next()` runs the rest of the chain once,
even if a wrap skips it: the SDK calls it right after a wrap that returns
or panics without calling it, and a second call does nothing. A wrap
returns `()`, so neither wrap can replace the user function's value or the
invocation's output.

`on_user_function_start` runs on a user function's first poll, and
`on_user_function_end` in the poll in which it returns, with
`Result<(), &WireError>`. The `WireError` that `on_user_function_end`
receives is the one the SDK records. Starts run in registration order, each
inside the earlier plugins' wraps, and ends run in reverse. The hooks are
interleaved into the wrap chain: a plugin's start runs before its own wrap,
and its end runs inside the earlier plugins' wraps after its own wrap
returned. That order of starts and ends applies to the earlier plugins
whose wraps called `next()`. After a wrap returns or panics before
`next()`, the rest of the chain runs outside it.

A start call that returns `Err` or panics skips that plugin's wrap and end
hook for the user function. This concerns the synchronous start call: after
it, the chain continues with the other plugins, and a future that the start
call returned and that fails later changes nothing. A user function that
panics, or that the SDK drops before it returns, gets no end hook, and an
operation replayed from its record fires neither hook. A future that a
start or end hook returns runs on its own task. The SDK never waits for it
on the operation's path; only the invocation-end drain waits for it.

Plugin code that claims an operation ID before the outcome is decided fails
the execution with a context usage error. The rule covers plugin code that
is active during an invocation, on the SDK's stack or on a hook task, and
the error names plugin code, as the error for a claim in a step body does.
For that rule, the outcome counts as decided at the driver's arbitration. A
builder claims its ID at the call, a deferred combinator at its first poll
or `.spawn()`, a `with_retry` block with its own record when the builder
becomes its future, and the first attempt of an unwrapped `with_retry`
block when that attempt starts. The plugin guard itself refuses nothing; admission and
task-ownership checks still apply, and the guard does not follow a builder
or future that the handler created, claimed and handed to the plugin. A
claim after the drive, in `on_invocation_end` or after `wrap_invocation`'s
`next()` during the end-of-invocation flush, records the error too. It
cannot shift an ID the execution uses, and it changes neither the response,
which comes from the drive's outcome, nor the late-panic precedence, which
comes from the driver's own arbitration result.

The user-function info of a child body takes `is_replay()` from the child
context when the body starts, so it can be `false` where
`on_operation_start` said `true`. A `ReplayChildren` reconstruction fires no
user-function hook.

No wrap may block, and the synchronous part of every hook must be cheap: a
wrap runs on whichever thread polls, as often as the SDK polls. A wrap
holds no lock across `next()` that its hooks take. That covers any
non-reentrant lock, such as a `std::sync::Mutex` guard or a `RefCell`
borrow, that one of the plugin's own hooks or wraps can take, because
`next()` runs nested user functions' hooks.

The end hook cannot inspect the typed successful value. `on_operation_end`
can inspect the serialized checkpoint payload through
`OperationInfo::result()` when the terminal update carries one; it is
`None` when that update carries no payload. An object-safe `Plugin` cannot
expose a method generic over each operation's `O`.

Example 2 puts a trace ID on every log event of a user function:
`wrap_user_function` calls `span.in_scope(next)`, so the events the user
function emits are parented to the span and carry its `trace_id` field.

```rust no_run
use aws_durable_execution_sdk as durable;
use durable::plugin::{InvocationInfo, OperationInfo, Plugin, PluginFactory};

struct TraceIdFactory;

impl PluginFactory for TraceIdFactory {
    fn create_plugin(
        &self,
        info: &InvocationInfo,
    ) -> Result<Option<Box<dyn Plugin>>, durable::BoxError> {
        let root = info.x_ray_trace_id().unwrap_or("no-trace").to_owned();
        Ok(Some(Box::new(TraceIdPlugin { root })))
    }
}

/// Holds the invocation's X-Ray trace header.
struct TraceIdPlugin {
    root: String,
}

impl Plugin for TraceIdPlugin {
    fn wrap_user_function(&self, info: &OperationInfo, next: &mut dyn FnMut()) {
        let id = format!("{}/{}", self.root, info.id());
        let span = tracing::info_span!("plugin_user_function", trace_id = %id);
        span.in_scope(next);
    }
}
```

### Context propagation

The SDK runs every task it starts for an operation (`.spawn()`, map items,
parallel and `parallel_mixed` branches, combinator inputs) inside the wraps
and the `tracing` span that were current when that operation's builder
became its future (`.await`, `.spawn()` or `.future()`), or, for a map item
or branch, when the container dispatched it, around every poll and every
drop. The handler's root task runs inside the invocation's wraps, captured
when it is dispatched. An operation awaited directly runs inside the wraps
of the code that awaits it, plus its own.

A user-started task is outside SDK propagation and must use the relevant
explicit carrier: tracing instrumentation, capture `Context::current()` and
call `.with_context(captured)` on the future for direct OpenTelemetry, or
the plugin's own mechanism.

With `tracing`, instrument the task's future with the current span:

```rust ignore
use tracing::Instrument as _;

tokio::spawn(work.instrument(tracing::Span::current()));
```

With OpenTelemetry used directly, capture the current OpenTelemetry context
and attach it to the task's future:

```rust ignore
use opentelemetry::context::FutureExt;

let captured = opentelemetry::Context::current();
tokio::spawn(work.with_context(captured));
```

With a plugin that keeps its context in an `Arc`, clone the `Arc` into the
spawned future and enter it through the function the plugin documents:

```rust ignore
let context = Arc::clone(&my_plugin::current_context());
tokio::spawn(async move { my_plugin::in_context(context, work).await });
```

### Log context

To add fields such as a trace ID to log events, enter a span that carries
the fields around `next()` in `wrap_invocation` or `wrap_user_function`.
Events emitted inside are parented to that span, and the application's
`tracing` subscriber or layer can render the span's fields. The SDK enters
the wraps on every task it starts for an operation.

### When each operation hook fires

"Start" is `on_operation_start`, "end" is `on_operation_end`, and "attempt
start" and "attempt end" are `on_attempt_start` and `on_attempt_end`. A
value in parentheses is the info's `is_replay`.

| Operation | No record (live) | Running record on replay | Live completion | Terminal record on replay |
| --- | --- | --- | --- | --- |
| `step` | start (`is_replay` false), attempt start | `STARTED`: start (`true`), attempt start. `READY`: start (`is_replay` = attempt > 1), attempt start. `PENDING`: nothing | success: attempt end, end. Retry: attempt end only. Final failure: attempt end, end | nothing |
| `wait_for_condition` | as `step` | as `step` | complete: attempt end, end. Continue: attempt end (`Succeeded`) only. Failure: attempt end, end | nothing |
| `wait` | start (`false`) | start (`true`) | end (`false`), when it succeeds while it waits in the invocation | `SUCCEEDED`: end, `is_replay` = not listed in `updated_operations`; any other status: nothing |
| `invoke` | start (`false`) | start (`true`) | end (`false`), when it reaches `SUCCEEDED`, `FAILED`, `TIMED_OUT`, `STOPPED` or `CANCELLED` while it waits in the invocation | `SUCCEEDED`, `FAILED`, `TIMED_OUT`, `STOPPED` or `CANCELLED`: end, `is_replay` = not listed in `updated_operations`; unknown status: nothing |
| `create_callback` | start (`false`) | start (`true`) | end (`false`), when `result()` reads `SUCCEEDED`, `FAILED` or `TIMED_OUT` after waiting in the invocation | `SUCCEEDED`, `FAILED` or `TIMED_OUT`: end, `is_replay` = not listed in `updated_operations` |
| `run_in_child_context`, `with_retry` with its own record, each `attempt-N` of a `with_retry` in either mode, `map`, `parallel`, `parallel_mixed`, each map item, parallel branch and `parallel_mixed` branch, `wait_for_callback`, the combinators | start (`false`) before the body | start (`true`) before the body | end (`false`) after the terminal write | nothing, including a `ReplayChildren` reconstruction |
| `run_in_child_context` with `virtual_context()`, and a virtual `with_retry` block | start (`is_replay` = the invocation's replay state) before the body, in every invocation | never, because a record at its ID fails replay | end (`is_replay` = the invocation's replay state) after the body returns, with no checkpoint write | never |

A step retry or a `wait_for_condition` check that continues in the same
invocation after its delay fires the `READY` cell of its row: start
(`true`), then attempt start. An item that a `map` or `parallel` abandons
because the batch completed early fires no end hook, nor do the unfinished
operations inside it. Each `NestingMode::Flat` map item, parallel branch or
`parallel_mixed` branch fires start and end hooks with `is_virtual` true and
the container as parent. It writes no record, and after abandonment, drop
or panic it fires no end hook. A `with_retry` block with
`.without_child_context()` has no record, so it fires no hook for itself;
its `attempt-N` children and waits fire theirs. The inner callback and step
of a named `wait_for_callback` report the names `<name>-callback` and
`<name>-submitter`.

### Info fields

`is_replay` is `false` when the SDK observed the event in this invocation,
and `true` on a step's second and later attempts, on re-entry of an
operation found running, and on the end hook of a wait, invoke or callback
replayed from a record the service did not change since the previous
invocation. `is_virtual` is `true` for a `NestingMode::Flat` map item or
parallel branch and for a virtual child context, which record no operation
of their own, and `false` for every other operation and every entry of
`operations()`. The start hook of a virtual child reports `is_replay`
`true` when an earlier invocation already ran its body, and its end hook
reports `true` when an earlier invocation already got past the child. `operations()`
holds every operation the service reports for this execution, including
the `EXECUTION` operation, keyed by wire ID.

`on_operation_change` is called once per checkpoint response that changed
at least one status, including the responses to the SDK's state refreshes
while operations wait, in response order, before `on_invocation_end`. Its
`updated_operations` is a subset of `operations`.

`propagation_trace_id` supplies the trace header of a chained invoke. The
first non-blank value across plugins becomes the `X-Amzn-Trace-Id` of a new
chained invoke; a replayed invoke does not ask.

`Options::builder().plugin_child_operations_depth(..)` keeps finished
children visible to plugins. By default the service drops the children of
a finished context from later invocations' state, so
`InvocationInfo::operations()` lists top-level operations only. The setting
keeps them to the given depth, at the replay cost its rustdoc states. A
preserved `wait_for_callback` re-runs its body on replay: its callback
fires its replay end hook, and its submitter step fires nothing.

### Isolation

A failing factory, hook, `name()`, wrap or hook work is logged as a WARN on
the target `aws_durable_execution_sdk::plugin` and changes nothing else. An
`Err` or a panic counts as a failure. The WARN sets `hook` to the method
name and names the plugin as `<Plugin::name()> (plugins[<i>])`, where
`<i>` is the 0-based registration index, or as `plugins[<i>]` with
`hook = "name"` when `name()` panicked. A factory's WARN carries
`factory = "<type_name::<F>()> (plugins[<i>])"` instead of `plugin` and
`hook`, plus `error` for a formatted `Err`. The one exception is a panic in
the destructor of an error that a hook or factory returned: it follows the
undelivered-panic path, as a panic in the SDK's own code does, so it fails
the invocation and the service invokes the function again. A binary built
with `panic = "abort"` aborts on a panicking hook.

### Plugins and `tracing`

Plugins and the `observability` module's spans and events are independent.
A plugin does not need a subscriber, and a subscriber does not need a
plugin. A plugin that wants the SDK's log records installs its own
`tracing` layer.

## Testing a handler locally

Enable the `test-util` feature to get `LocalRunner`, which drives a handler
through as many simulated invocations as the execution needs, in process and
without AWS. [Differences from the JavaScript SDK](docs/differences-from-js.md#testing)
lists where `LocalRunner` differs from the JavaScript SDK's local runner.

```rust ignore
let result = LocalRunner::new()
    .run(
        |n: u32, ctx: durable::DurableContext| async move {
            let v = ctx.step(move |_| async move { Ok(n + 1) }).await?;
            Ok::<_, durable::BoxError>(v)
        },
        41_u32,
    )
    .await?;
assert_eq!(result.output(), Some(&42));
```

`TestResult` also reports the invocation count and every recorded operation,
so a test can assert that a step ran once across replays.
`result.operation("charge")` returns the first operation with that name, and `TestOperation::error()` returns its full error record, including error data.
`TestOperation::kind()` and `TestOperation::status()` return `OperationKind`
and `OperationStatus`, so a test checks
`op.status() == OperationStatus::Succeeded`.
`println!("{result}")` prints the operations as a table, and `result.display_with(columns)` selects the columns.
`result.history_events()` returns the execution's history events, and `result.invocations()` lists each handler invocation with its request ID and error.
Before it awaits the run, a test can wait for an operation with `run.operation(name).wait_for_data(..)` and answer its callback with `send_callback`, as the `test_util` docs show under "Reacting to a run in progress".

`.register_function(name, f)` and `.register_durable_function(name, handler)`
supply the target of `ctx.invoke`.

A test queues a reply for a callback before the run.
`.callback(CallbackReply::...)` queues one for the next pending callback, in
order, and `.callback_for(name, CallbackReply::...)` queues one for the
callback with that name, or for the `wait_for_callback` with that name.
`CallbackReply::success(&value)` delivers `value` encoded as JSON and returns
the encoding error when `value` fails to serialize. `CallbackReply::raw`
delivers a payload unchanged, `CallbackReply::failure` delivers an external
failure built from a `CallbackFailure`, and `CallbackReply::timeout` and
`CallbackReply::heartbeat_timeout` deliver the records a callback timeout and
a missed heartbeat produce. The runner runs no callback timers itself.

```rust ignore
use durable::test_util::{CallbackFailure, CallbackReply, LocalRunner};

let runner = LocalRunner::new()
    .callback_for("approval", CallbackReply::success(&true)?)
    .callback(CallbackReply::failure(CallbackFailure::new("Declined", "no")));
```

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

## Acknowledgments

Big thanks to [@jlizen](https://github.com/jlizen),
[@landonxjames](https://github.com/landonxjames), and
[@bnusunny](https://github.com/bnusunny), whose insightful advice on the
early experimental versions caught the bugs before they became bugs, and
whose eagle-eyed reviews caught the subtle (and not-so-subtle!) ones before
they could bite anyone.

## License

Apache-2.0. See [LICENSE](LICENSE).
