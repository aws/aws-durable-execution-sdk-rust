# Differences from the JavaScript SDK

This page lists every place where the Rust SDK behaves differently from the JavaScript SDK. Each entry gives the JavaScript behavior, the Rust behavior, the reason, the decision behind it, and where the Rust behavior lives.

## Runtime and transport

### User-Agent

- **JavaScript:** Appends `-bundled` to the version only when the Lambda runtime ships the SDK.
- **Rust:** Never appends a suffix.
- **Why:** No Lambda runtime bundles the Rust SDK, so every Rust function is the JavaScript case that sends no suffix.
- **Decision:** Same capability in Rust form.
- **Where:** `src/client.rs`, `USER_AGENT_TOKEN` and `UserAgentTokenInterceptor`.

### Paginated checkpoint response

- **JavaScript:** Applies the response's operations and ignores its `NextMarker`, so it misses an operation the service paged out.
- **Rust:** Fetches the full state with `GetDurableExecutionState` and merges it.
- **Why:** A paged-out operation is never seen in JavaScript.
- **Decision:** JavaScript defect and Decision (2026-08-09, [Rust SDK #5: Bootstrap path does not paginate execution state; long histories replay against a truncated log](https://github.com/aws/aws-durable-execution-sdk-rust/issues/5)): fetch the full execution state.
- **Where:** `src/context.rs`, `checkpoint_direct_with_events`.

### SDK debug output

- **JavaScript:** Printed when `DURABLE_VERBOSE_MODE=true`.
- **Rust:** Emitted on the `tracing` target `aws_durable_execution_sdk::lifecycle` at DEBUG; `RUST_LOG` selects it.
- **Why:** A `tracing` target is the Rust way to opt in to verbose output.
- **Decision:** Same capability in Rust form.
- **Where:** `src/observability.rs`, `TARGET`.

### Operation IDs

- **JavaScript:** First 16 hex characters of the MD5 of the positional ID.
- **Rust:** Full SHA-256 hex of the positional ID.
- **Why:** The IDs are opaque strings; no API or service behavior compares them across SDKs.
- **Decision:** Decision (2026-10-31): keep the full SHA-256 hex.
- **Where:** `src/engine.rs`, `compute_wire_id`.

### What can suspend

- **JavaScript:** The invocation ends when no operation runs and some operation waits, after a 3 ms cooldown. So a handler `await` longer than 3 ms between operations can let it suspend.
- **Rust:** It suspends, with no cooldown, as soon as no SDK-owned work can make progress and some operation waits; a step body, condition check or serdes call holds an activity permit, which the owning combinator, a closed scope or the end of the drive can stop, and an untracked `await` keeps nothing open. A handler's return ends the invocation and aborts an operation behind a dropped `.spawn()` handle, whose record stays `STARTED`, so the execution can succeed with it unfinished, as in JavaScript.
- **Why:** In JavaScript the same handler finishes or suspends depending on whether an untracked `await` took 2 ms or 4 ms. Rust tracks every task it starts and every result not yet consumed, so it sees an exact zero and needs no cooldown.
- **Decision:** JavaScript defect and Decision (2026-10-31): suspend exactly when no tracked work can make progress and an operation waits.
- **Where:** `src/coordinator.rs`, `InvocationCoordinator`; `src/context.rs`, `DurableContext::hold_activity`.

### Work left when the drive ends

- **JavaScript:** Drains the queued checkpoints and returns; work still running keeps running.
- **Rust:** Aborts and awaits the handler's root task and every tracked operation task, including one whose `.spawn()` handle was dropped. It then waits for work it cannot abort, such as admitted checkpoint writes, drains plugin hook work, and returns.
- **Why:** Lambda resumes unfinished background work when it reuses the execution environment ([Lambda Developer Guide](https://docs.aws.amazon.com/lambda/latest/dg/lambda-runtime-environment.html)). So JavaScript's leftover work can run during a later invocation and checkpoint through the earlier invocation's checkpoint manager.
- **Decision:** JavaScript defect.
- **Where:** `src/driver.rs`, `drive_invocation`; `src/plugin.rs`, `InvocationPlugins::invocation_end`.

### Custom checkpoint transport

- **JavaScript:** `DurableExecutionClient` and `DurableExecutionClientError`, with scopes `INVOCATION` and `EXECUTION` and an optional `cause`; the default transport is `DurableExecutionApiClient`.
- **Rust:** The `ExecutionClient` trait, `ExecutionClientError` with an `ExecutionClientErrorScope` and an optional source, and the default transport `LambdaExecutionClient`. `get_state` reads one page, and a tokenless checkpoint response is a `None` token.
- **Why:** A trait is the Rust form of an interface.
- **Decision:** Same capability in Rust form.
- **Where:** `src/client.rs`, `ExecutionClient`.

## Logging

### Replay suppression

- **JavaScript:** `ctx.logger` drops records during replay, and `configureLogger({ modeAware })` changes that per context.
- **Rust:** `init_default_subscriber()` and `init_default_subscriber_with_writer(writer)` install a subscriber filtered by `ReplayFilterLayer`. A per-target filter keeps chosen records during replay.
- **Why:** In Rust the subscriber owns filtering, and `tracing` already filters per target.
- **Decision:** Same capability in Rust form.
- **Where:** `src/tracing_layer.rs`, `ReplayFilterLayer`.

### SDK logger

- **JavaScript:** The context carries a logger, and a plugin's `enrichLogContext` adds fields to its records.
- **Rust:** No SDK logger. A plugin enters a span carrying the fields around `next()` in `wrap_invocation` or `wrap_user_function`; events inside are parented to it, and the subscriber can render its fields.
- **Why:** A `tracing` span is the standard way to attach fields to every record in a piece of code.
- **Decision:** Same capability in Rust form.
- **Where:** `src/plugin.rs`, `Plugin::wrap_user_function`.

## Errors

### Untyped handler error

- **JavaScript:** Records the error's `name` as the type, so a plain `Error` records `Error`, and the error's message. A result that `JSON.stringify` cannot serialize records the thrown error's name, such as `TypeError`.
- **Rust:** Records the type `Error` and the flattened error chain as the message; `TypedError` sets another type. A result that fails to serialize records `Error` and the `serde_json` error message.
- **Why:** A boxed Rust error and a `serde_json::Error` have no name, so they record the type a plain JavaScript `Error` records.
- **Decision:** Rust language.
- **Where:** `src/lib.rs`, `wire_error_from_box_error` and `serialize_handler_result`.

### Error type strings

- **JavaScript:** Rebuilds an error class from the recorded type; an unknown type becomes `StepError`.
- **Rust:** Keeps the operation's own error kind and carries the recorded type as data in `OperationError::wire()`.
- **Why:** An enum kind plus the type string carries what `instanceof` plus `errorType` carries.
- **Decision:** Same capability in Rust form.
- **Where:** `src/error.rs`, `WireError`.

### Stack traces

- **JavaScript:** Records none.
- **Rust:** Records none of its own, but keeps frames that a record already carries, such as a remote invoke's.
- **Why:** Frames from another system are diagnostic data the SDK did not create.
- **Decision:** Decision (2026-10-28): capture no frames, and keep the frames a record already carries.
- **Where:** `src/error.rs`, `wire_error_with_type`; `src/invoke.rs`, `invoke_error_from_record`.

### Child failure type

- **JavaScript:** A child whose body propagates a failed step records `StepError`.
- **Rust:** Records the body error's type. For an operation error the body propagates, that is the type the inner operation recorded, else the type of a `TypedError` in its source chain, else that error kind's name, such as `StepError`. For a replayed failure the body propagates, it is the type that failure's record carries, else `Error`. For any other error, it is the type of a `TypedError` in the source chain, else `Error`.
- **Why:** The inner type tells a history reader more, and the Rust error kind `ChildContext(ChildFailed)` still says that the child failed.
- **Decision:** Decision (2026-10-28): record the body's error type.
- **Where:** `src/error.rs`, `ChildFnError::failure_wire` and `wire_error_for`; `src/child.rs`, `checkpoint_live_failure`.

### Panic in user code

- **JavaScript:** Not applicable: a thrown error is an operation failure.
- **Rust:** A panic in a handler, step body, condition check, child body, map item, parallel branch or combinator input ends the invocation as a Lambda error with its original payload and records nothing, so a panic that recurs on every invocation retries until the execution times out. A panic in an SDK task whose result handle or join set was dropped also fails the invocation, and an `Err` stays an operation failure.
- **Why:** A panic is a bug, not a durable outcome; recording it would put a crash in the history. The task gateway records an invocation-failing panic even when nothing reads its outcome, and the invocation coordinator keeps the first one, so the driver still fails the invocation.
- **Decision:** Decision (2026-10-28): propagate a panic its consumer observes, and record no panic.
- **Where:** `src/task.rs`, `record_or_discard` and `TaskTracker::claim_unread_panics`; `src/coordinator.rs`, `InvocationCoordinator::record_panic`.

### Condition strategy exhaustion

- **JavaScript:** Throws a plain error `waitForCondition exceeded maximum attempts (N)`, which records `Error`; replay rebuilds an untyped error.
- **Rust:** Fails with `WaitForCondition(MaxChecksExceeded)` carrying `checks()`, recorded as `WaitForCondition.MaxChecksExceeded`, live and on replay.
- **Why:** The handler can catch this failure, so replay must rebuild the same matchable kind from the record, which `Error` would hide among check failures; Python does the same. A failure that ends the execution uncaught, such as a wrong-context call, records `Error` as JavaScript does.
- **Decision:** Decision (2026-10-31): record the typed `WaitForCondition.MaxChecksExceeded`.
- **Where:** `src/wait_for_condition.rs`, `WaitDecision::Exhausted`.

### `wait_for_callback` failure records

- **JavaScript:** The outer record carries the error class name: `CallbackExternalError`, `CallbackSubmitterError` or `CallbackTimeoutError`.
- **Rust:** The outer record carries one of `Callback.ExternalFailure`, `Callback.SubmitterFailed`, `Callback.Timeout`, `Callback.Heartbeat` and `Callback.Internal`; an external failure keeps the external message and error data and drops the external type, as in JavaScript. A payload that does not decode returns `CallbackErrorKind::DeserializationFailed` after the outer record succeeded with the raw payload.
- **Why:** A fixed type set is a closed protocol the SDK owns; each type maps to one error kind, live and on replay.
- **Decision:** Decision (2026-10-28): record one of a fixed set of `Callback.*` types.
- **Where:** `src/callback.rs`, `extract_wire_error`.

### Callback timeout kinds

- **JavaScript:** Every `TIMED_OUT` callback record becomes `CallbackTimeoutError`.
- **Rust:** A `TIMED_OUT` record whose error type is `Callback.Heartbeat` becomes `CallbackErrorKind::HeartbeatTimedOut`; every other `TIMED_OUT` becomes `TimedOut`.
- **Why:** A handler can tell a missed heartbeat from an overall timeout.
- **Decision:** Decision (2026-10-28): report a heartbeat timeout as its own kind.
- **Where:** `src/callback.rs`, `timed_out_callback_error` and `classify_callback_error`.

### Serdes failure message

- **JavaScript:** `Serialization failed for step "<name>" (<id>): <message>`, and `Deserialization failed ...` for a decode; without a name, `for step (<id>)`.
- **Rust:** For a failure marked `RetryableSerdesError`, the invocation error is `serdes serialize failed for operation "<name>" (<id>): <error chain>`, and `serdes deserialize failed ...` for a decode. Without a name it reads `for operation <id>`.
- **Why:** The message carries the same name, ID and cause in the crate's lowercase error style; nothing parses the text.
- **Decision:** Same capability in Rust form.
- **Where:** `src/context.rs`, `DurableContext::serdes_fault`.

## Operations

### Custom retry and poll delays

- **JavaScript:** Sent as given, so a zero delay is sent as 0.
- **Rust:** A caller-given delay is rounded up to whole seconds, at least 1 s. A computed strategy delay rounds to the nearest second with a 1 s floor, as in JavaScript.
- **Why:** The service counts whole seconds and rejects 0. A caller's duration is a minimum, so rounding up never fires early.
- **Decision:** JavaScript defect and Decision (2026-10-28): round a caller delay up and a strategy delay to the nearest second, with a 1 s minimum.
- **Where:** `src/step.rs`, `quantize_delay`, `RetryStrategyConfig::decide` and `handle_failure`; `src/builders/wait_for_condition.rs`, `WaitStrategy::decide`; `src/wait_for_condition.rs`, `WfcCore::settle`.

### `ctx.wait` duration

- **JavaScript:** Sends the raw seconds; a zero wait is sent as 0 and the service rejects it.
- **Rust:** Rounds up to whole seconds and records at least 1 s.
- **Why:** A requested wait is a minimum, and 0 is not a valid wait.
- **Decision:** JavaScript defect and Decision (2026-10-28): round a wait up to whole seconds, with a 1 s minimum.
- **Where:** `src/context.rs`, `DurableContext::wait`.

### Wait record with an unexpected status

- **JavaScript:** Resolves the wait for `CANCELLED`, `FAILED`, `STOPPED` and `TIMED_OUT`, after logging "Wait ended with unexpected status".
- **Rust:** Returns `Wait(UnexpectedStatus)` for every status other than `STARTED` and `SUCCEEDED`.
- **Why:** In JavaScript, code after a cancelled wait runs as if the timer elapsed.
- **Decision:** JavaScript defect and Decision (2026-10-31): return `Wait(UnexpectedStatus)`.
- **Where:** `src/wait.rs`, `recorded_outcome`, called from `WaitExecution::execute_after_start`.

### `wait_for_condition` replay check

- **JavaScript:** Skips the replay identity check that every other operation makes.
- **Rust:** Checks type, sub-type and name like every operation, and fails a mismatch with `NonDeterministicExecutionError`.
- **Why:** In JavaScript, a changed handler that put a step at a condition's position reads that step's record as the condition's state.
- **Decision:** JavaScript defect.
- **Where:** `src/wait_for_condition.rs`, `WfcCore::before`.

### `wait_for_condition` shape

- **JavaScript:** `initialState` is required, checked at run time. The strategy returns `{ shouldContinue }`, true to keep polling.
- **Rust:** A typed initial state is required, checked at compile time. The strategy returns `WaitDecision`, and `WaitStrategy::builder(predicate)` takes a predicate that returns `true` when done.
- **Why:** Types replace a run-time check; the predicate states the same condition with the opposite sign.
- **Decision:** Same capability in Rust form.
- **Where:** `src/context.rs`, `wait_for_condition`.

### `WaitStrategyConfig.timeoutSeconds`

- **JavaScript:** Declared and never read.
- **Rust:** Absent.
- **Why:** A JavaScript user gets no timeout from it either.
- **Decision:** Same capability in Rust form.
- **Where:** `src/builders/wait_for_condition.rs`, `WaitStrategy`.

### Invoke input

- **JavaScript:** An omitted input sends no payload, so a JavaScript target reads `{}`.
- **Rust:** The input is required and typed; `()` sends `null`, so a JavaScript target reads `null`. `serde_json::json!({})` delivers what JavaScript's omitted input delivers.
- **Why:** A required typed input avoids JavaScript's overload misread, and a deliberate `null` stays distinct from no input.
- **Decision:** Decision (2026-10-31): keep the required typed input.
- **Where:** `src/invoke.rs`, `serialize_invoke_input`.

### Invoke arguments

- **JavaScript:** A positional overload can read a string input as the function name.
- **Rust:** `invoke(function_id, input)` with typed parameters.
- **Why:** In JavaScript, `invoke("fn-a", "some-string")` sends to function `some-string`.
- **Decision:** JavaScript defect.
- **Where:** `src/context.rs`, `invoke`.

### `with_retry` structure

- **JavaScript:** Runs every attempt directly in the one child context of the `withRetry` operation, and names the waits `{name}-backoff-{attempt}` when the block is named, and leaves them unnamed otherwise. `wrapWithRunInChildContext: false` runs the attempts in the caller's context, and `virtualContext` writes no block record.
- **Rust:** Runs each attempt in its own child context `attempt-N` inside the `with_retry` child context; `without_child_context()` runs the attempts in the caller's context, `virtual_context()` writes no block record, and both name a named block's attempt children `{name}-attempt-N`. In every mode a named block's waits are `{name}-backoff-N`, where `N` is the failed attempt, and an unnamed block's waits are unnamed, as in JavaScript.
- **Why:** Replay returns a failed attempt's recorded failure without calling its closure, so the strategy sees the same error on every invocation, and the next attempt's operation IDs do not depend on what a failed attempt created. Each attempt costs one record.
- **Decision:** Decision (2026-10-28 and 2026-10-31): one child context per attempt, and a named unwrapped or virtual block prefixes its attempt children's names.
- **Where:** `src/with_retry.rs`, `retry_loop`; `src/builders/with_retry.rs`, `WithRetryBuilder`.

### Step closure per attempt

- **JavaScript:** Calls the same step function object for each attempt in one invocation, so a capture that attempt 1 changed stays changed for attempt 2.
- **Rust:** Calls a fresh clone of the original `FnOnce + Clone` closure for each attempt. Each capture follows its type's `Clone` implementation: a `Vec` is copied, an `Arc` is shared.
- **Why:** Calling an `FnOnce` consumes it, so a second call needs a clone. In JavaScript an attempt's local state depends on whether the retry ran in the same invocation, because a re-invoked handler builds the closure again.
- **Decision:** Rust language and Decision (2026-10-26): call a fresh clone of the closure for each attempt.
- **Where:** `src/step.rs`, `StepExecution::run`.

### Custom subtype

- **JavaScript:** Any string; an empty string selects the default, and the service rejects an invalid one.
- **Rust:** `OperationSubType::new` accepts 1 to 32 characters from `A-Z`, `a-z`, `0-9`, `-` and `_`, and rejects an empty value.
- **Why:** An invalid subtype cannot reach a builder; not calling `.sub_type` is the default.
- **Decision:** Decision (2026-10-28): validate the subtype when it is built.
- **Where:** `src/operation_sub_type.rs`, `OperationSubType::new`.

### Callback payload decoding

- **JavaScript:** Returns the raw string by default; `defaultCallbackDeserializer` sets an execution-wide decoder.
- **Rust:** Decodes with `JsonSerdes` by default, and `.serdes(PassthroughSerdes)` returns the raw string. There is no execution-wide decoder.
- **Why:** A builder's default serdes type cannot depend on the output type, so a raw-string default would type-check only for `String`.
- **Decision:** Decision (2026-08-20, [Rust SDK #37: Redesign Serdes with typed values and implementation-controlled async work](https://github.com/aws/aws-durable-execution-sdk-rust/issues/37)): decode with the operation's typed serdes, `JsonSerdes` by default.
- **Where:** `src/builders/callback.rs`, `CreateCallbackBuilder` and `WaitForCallbackBuilder`; `src/serdes.rs`, `PassthroughSerdes`.

### Absent callback payload

- **JavaScript:** A callback that succeeded without a payload gives the default pass-through deserializer `undefined`, which a literal `null` payload does not give.
- **Rust:** An absent payload reaches the serdes as `"null"`, so `PassthroughSerdes` returns the string `"null"`, as for a literal `null` payload. `JsonSerdes`, the default, turns `"null"` into `None` for an `Option` output.
- **Why:** Rust has no `undefined`, and `Serdes::deserialize` takes a `String` for every operation. An `Option<String>` argument would change every user serdes.
- **Decision:** Rust language and Decision (2026-10-31): pass `"null"` for an absent payload and keep `Serdes::deserialize` taking a `String`.
- **Where:** `src/callback.rs`, `callback_outcome` and `WfcbCore::before`.

### `wait_for_callback` record

- **JavaScript:** Without `defaultCallbackDeserializer`, the outer context records `JSON.stringify(raw)`.
- **Rust:** The outer context records the raw payload string, and the configured serdes decodes it once, live and on replay.
- **Why:** One decode path, live and on replay, matching JavaScript's configured path.
- **Decision:** Decision (2026-10-28): record the raw payload and decode it once.
- **Where:** `src/callback.rs`, `WaitForCallbackExecution::run` and `WfcbAfter::settle`.

### Child error mapping

- **JavaScript:** `ChildConfig.errorMapper`.
- **Rust:** No option; call `map_err` on the awaited result.
- **Why:** `Result::map_err` after `.await` produces the same error with the same record.
- **Decision:** Same capability in Rust form.
- **Where:** `src/context.rs`, `run_in_child_context`.

### Record at a virtual child's slot

- **JavaScript:** Returns the record and skips the virtual body.
- **Rust:** Fails with `NonDeterministicExecutionError`, as Python does.
- **Why:** A virtual child writes no record, so a record there comes from an earlier handler version, and JavaScript returns that old result.
- **Decision:** JavaScript defect.
- **Where:** `src/context.rs`, `DurableContext::reject_record_at_virtual_child`.

### Unpolled `.future()`

- **JavaScript:** An ordinary operation starts at its call, so a step that nothing awaits still runs and records its result. A combinator is lazy.
- **Rust:** `.await` and `.spawn()` start an operation at once, and an ordinary `.future()` starts at its first poll, so one that nothing polls records nothing but has claimed its ID, and later IDs match JavaScript. A `with_retry` block with `.without_child_context()` claims no ID until its first attempt starts. A combinator is lazy, as in JavaScript, and claims no ID until its first poll or `.spawn()`.
- **Why:** A combinator starts the `.future()` inputs it is given, so it can stop the ones it does not use. A replay that returns its recorded result starts none of them.
- **Decision:** Decision (2026-11-03, [Rust SDK #151: Keep dropped spawned operations running, and stop work when its scope closes](https://github.com/aws/aws-durable-execution-sdk-rust/issues/151)): start a `.future()` at its first poll.
- **Where:** `src/future.rs`, `DurableFuture::for_start_step` and `DurableFuture::deferred`.

## Batches and combinators

### Exceeded failure tolerance

- **JavaScript:** `map` always returns a batch result, and `throwIfError()` turns a failure into an error.
- **Rust:** `map(..).await` returns the succeeded values, or `Err` when the failure tolerance is exceeded or the completion predicate returns `Complete(Failed)` (the first failed item's error, else `ChildContext(BatchCompletionFailed)`). `await_batch()` returns the batch result as JavaScript's `map` does, `BatchResult::into_result()` is `throwIfError()`, and the parent record is `SUCCEEDED` in both.
- **Why:** `await_batch()` plus `into_result()` is JavaScript's `map` plus `throwIfError()`. `.await` is the shorter form for the common case and returns `Err` only when the batch failed.
- **Decision:** Same capability in Rust form.
- **Where:** `src/builders/map_parallel.rs`, `await_batch`; `src/map_parallel.rs`, `BatchResult::into_result`.

### Batch record payload

- **JavaScript:** `{"all":[...],"completionReason":...}` embeds each item's value as JSON, also for a custom `itemSerdes`, and each error as a recursive error object. An oversized batch writes the `MapResult` or `ParallelResult` summary.
- **Rust:** The same keys and summary; a JSON item embeds the exact JSON text its item serdes wrote, and a custom-serdes item stores that serdes's string as a JSON string. An item error is a one-cause form of JavaScript's error object.
- **Why:** Generic Rust code cannot test whether a custom-serdes item type implements `Serialize`. An item type may have no JSON form.
- **Decision:** Rust language.
- **Where:** `src/map_parallel.rs`, `from_batch_result`.

### Batch item names

- **JavaScript:** The batch record items and `BatchResult` items carry no name.
- **Rust:** Every record item, `BatchItem` and `BatchError` carries the recorded name, which survives replay.
- **Why:** Structured per-item errors carry the item index, the operation name and the error value.
- **Decision:** Decision (2026-08-09, [Rust SDK #15: BatchResult: status should be an enum; errors() loses error identity and item association](https://github.com/aws/aws-durable-execution-sdk-rust/issues/15)): carry the recorded name on every item and item error.
- **Where:** `src/map_parallel.rs`, `BatchItem`.

### Failure percentage arithmetic

- **JavaScript:** Fails the batch when `(failureCount / items.length) * 100 > tolerated`, so 7 failures of 50 at 14 fail it (`14.000000000000002 > 14`).
- **Rust:** Fails the batch when `failure_count * 100 > pct * total_items` in `f64`, so 7 failures of 50 at 14 do not fail it. A fraction such as `12.5` is accepted.
- **Why:** The JavaScript division rounds up past the tolerance.
- **Decision:** JavaScript defect and Decision (2026-08-06, [Rust SDK #19: tolerated_failure_percentage: integer-division boundary behavior and ParallelConfig percentage support need a spec decision](https://github.com/aws/aws-durable-execution-sdk-rust/issues/19)): compare by cross-multiplication.
- **Where:** `src/map_parallel.rs`, `should_stop_failure`, which `CompletionTracker::settle` calls.

### Items running at early completion

- **JavaScript:** The items keep running after the batch completes. JavaScript drops their later checkpoints, and their callers never resume.
- **Rust:** The SDK aborts each running item's task, so its future stops at its next `.await` and work in the body that is not a step can stop part way. The item's operation and plugin end hooks never fire and its record stays `STARTED`, so durable side effects belong in steps.
- **Why:** Rust cancels a future by dropping it, so the SDK stops work whose checkpoints would be rejected. An abandoned item's writes are rejected at admission, as JavaScript drops them.
- **Decision:** Decision (2026-10-26): abort the running items.
- **Where:** `src/map_parallel.rs`, `produce_batch`.

### Combinator failure records

- **JavaScript:** Records the type `PromiseCombinatorError`, and replay rebuilds an untyped error.
- **Rust:** Records structured discriminators that replay to the same error kind and index.
- **Why:** The handler can catch these failures, so replay must rebuild the same error kind from the record.
- **Decision:** Decision (2026-10-31): record the combinator discriminators.
- **Where:** `src/combinator.rs`, `join_failed_wire`, `first_failure_wire` and `replay_combinator_failure`.

### `race([])`

- **JavaScript:** Never settles.
- **Rust:** Fails at once with `EmptyInput`.
- **Why:** In JavaScript, a handler that races an empty list hangs until the execution times out.
- **Decision:** JavaScript defect.
- **Where:** `src/combinator.rs`, `RaceExecution::execute`.

### Unused combinator input

- **JavaScript:** Keeps running.
- **Rust:** When `race`, `select_ok` or `try_join_all` resolves, each `.future()` input still running is cancelled at its next `.await`. A `.spawn()` input keeps running, as in JavaScript, and `join_all` cancels nothing.
- **Why:** A loser's work is not used, and Rust cancels a future by dropping it. On replay a `.spawn()` input starts again before the combinator exists, so stopping it could leave its record `STARTED` and run a step body again.
- **Decision:** Decision (2026-10-26): cancel the unused `.future()` inputs.
- **Where:** `src/combinator.rs`, `RaceExecution::execute`, `SelectOkExecution::execute` and `TryJoinAllExecution::execute`.

### Stopped work and its records

- **JavaScript:** `race` and `any` do not stop their losers: a running loser step keeps the invocation from suspending and can finish, and a loser that waits on the service can stay `STARTED` while the execution completes. Items still running when a `map` or `parallel` completes early run until their next checkpoint, which JavaScript refuses.
- **Rust:** A combinator stops its unsettled `.future()` inputs, and a closed child context, `map`, `parallel` or `parallel_mixed` stops the tasks still running inside it, nested work included, at their next `.await`. The records of that work can stay `STARTED`.
- **Why:** Rust can stop a task the SDK started, while JavaScript and Python cannot, and their refusal of late checkpoints is best effort. Stopping ends the invocation sooner and limits the side effects.
- **Decision:** Decision (2026-10-26 and 2026-11-03, [Rust SDK #151: Keep dropped spawned operations running, and stop work when its scope closes](https://github.com/aws/aws-durable-execution-sdk-rust/issues/151)): stop combinator `.future()` losers and the work of a closed scope.
- **Where:** `src/combinator.rs`, `RaceExecution::execute`, `SelectOkExecution::execute` and `TryJoinAllExecution::execute`; `src/child.rs`, `ChildExecution::run`; `src/checkpoint_coalescer.rs`, `ScopeClose`; `src/map_parallel.rs`, `produce_batch`.

### Oversized `race` or `select_ok` result

- **JavaScript:** Replay runs the combinator again and can return a different winner.
- **Rust:** Records `{"winner":i}` with `ReplayChildren`, so replay drives only the recorded winner and returns the same value.
- **Why:** In JavaScript, replay can return a value the live run did not return.
- **Decision:** JavaScript defect.
- **Where:** `src/combinator.rs`, `checkpoint_winner_succeed` and `replay_recorded_winner`.

### Batch serdes names

- **JavaScript:** `itemSerdes` and `serdes`.
- **Rust:** `serdes` (items) and `result_serdes` (the batch result).
- **Why:** Matches the per-operation `serdes` of every other builder.
- **Decision:** Same capability in Rust form.
- **Where:** `src/builders/map_parallel.rs`, `MapBuilder::serdes`, `ParallelBuilder::serdes` and the `result_serdes` method of each batch builder.

### Whole-batch serdes input

- **JavaScript:** `serdes` receives the `BatchResult` object.
- **Rust:** `result_serdes` is a `Serdes<String>` that receives the batch record's JSON text. The default `PassthroughSerdes` stores it unchanged.
- **Why:** A typed `BatchResult<O>` would require `O: Serialize + DeserializeOwned` of every map, or would run each item serdes twice. A lossless string transform stores the record without interpreting item values.
- **Decision:** Same capability in Rust form.
- **Where:** `src/builders/map_parallel.rs`, `MapBuilder::result_serdes`, `ParallelBuilder::result_serdes` and `ParallelMixedBuilder::result_serdes`.

## Serialization

### Serdes failures

- **JavaScript:** Every serialize or deserialize failure ends the invocation, records nothing, and the service invokes the function again.
- **Rust:** A failure is permanent unless the serdes wraps it in `RetryableSerdesError`, which ends the invocation and records nothing, as in JavaScript; `FileSystemSerdes` marks its filesystem I/O errors and an exhausted unique-name search, and leaves its JSON, envelope and `base_path` containment errors unmarked. A permanent serialize failure after a body ran, or a permanent failure of the state round trip after a live wait-for-condition check, records `FAIL` with the error type `SerializationError`, so the body runs once; a permanent failure of the initial-state round trip, which runs before `START`, returns `SerializationFailed`, runs the check zero times and writes no update; the error kind is `SerializationFailed` for a step, an invoke or a wait-for-condition and `ChildContext(Internal)` for a child context. A `Flat` batch item records the failure as a failed item in the batch record, and a failed `result_serdes` serialize records nothing. A permanent deserialize failure of a recorded value returns a catchable error and records nothing.
- **Why:** In JavaScript a permanent defect re-runs a body on every invocation until the execution times out. Python and Go make the same split.
- **Decision:** JavaScript defect and Decision (2026-10-28): a serdes failure is permanent unless the serdes marks it retryable.
- **Where:** `src/serdes.rs`, `RetryableSerdesError` and `is_retryable_serdes_error`; `src/context.rs`, `DurableContext::route_serdes_error`.

### Default serdes

- **JavaScript:** `configureSerdes({ defaultSerdes })` sets one serdes for every operation of a context.
- **Rust:** Typed serdes per operation; share one with `Arc<S>`.
- **Why:** A single slot cannot hold `Serdes<T>` for every output type without erasing the value again.
- **Decision:** Rust language and Decision (2026-08-20, [Rust SDK #37: Redesign Serdes with typed values and implementation-controlled async work](https://github.com/aws/aws-durable-execution-sdk-rust/issues/37)): a typed serdes per operation.
- **Where:** `src/serdes.rs`, `Serdes`.

### `FileSystemSerdes`

- **JavaScript:** Reuses one file name per operation and reads any file reference.
- **Rust:** Writes unique immutable files and honors file references only from SDK-written records under `base_path`.
- **Why:** In JavaScript a retry overwrites a file an accepted checkpoint references, and external input can point at any local file.
- **Decision:** JavaScript defect and Decision (2026-08-21, [Rust SDK #46: FileSystemSerdes: writes mutate state referenced by accepted checkpoints](https://github.com/aws/aws-durable-execution-sdk-rust/issues/46)): write unique immutable files, and honor only SDK-written file references.
- **Where:** `src/serdes.rs`, `FileSystemSerdes`, `PayloadOrigin`.

### Preview hook

- **JavaScript:** `generatePreview` receives the value and runs after the file is written. A hook that throws leaves the file on disk unreferenced.
- **Rust:** `FileSystemSerdesConfigBuilder::preview` receives the value's JSON text and runs before the file is written. It returns `Result`, and an `Err` fails the serialize and writes no file.
- **Why:** The config of a blanket `Serdes<T>` impl cannot name `T`, and `serde_json` stays off the public API. JavaScript's orphan payload file is a defect.
- **Decision:** JavaScript defect and Decision (2026-08-20, [Rust SDK #37: Redesign Serdes with typed values and implementation-controlled async work](https://github.com/aws/aws-durable-execution-sdk-rust/issues/37)): a hook over the JSON text that runs before the file is written.
- **Where:** `src/serdes.rs`, `FileSystemSerdesConfigBuilder::preview`.

## Plugins

### Registration

- **JavaScript:** Takes factories in `config.plugins`, and also imports each module named in `DURABLE_EXECUTION_PLUGINS`. An invalid provider fails the invocation with `PluginLoadError`.
- **Rust:** Typed factories registered with `OptionsBuilder::plugin`; no environment loading.
- **Why:** A Rust binary links statically and cannot load a crate at run time. The load-time checks become compile errors.
- **Decision:** Rust language and Decision (2026-10-31): register typed factories only.
- **Where:** `src/options.rs`, `OptionsBuilder::plugin`.

### Exclusive groups

- **JavaScript:** Rejects two factories with the same non-empty group at load, and every invocation fails with `PluginLoadError`.
- **Rust:** `OptionsBuilder::build` returns `OptionsValidationError`. An empty group is no group, as in JavaScript.
- **Why:** A builder that validates its configuration is the Rust place, and the user gets the same rejection before any invocation.
- **Decision:** Same capability in Rust form.
- **Where:** `src/options.rs`, `OptionsBuilder::build`.

### Wrap shape

- **JavaScript:** One call `wrap(info, fn)` per user function, nested with the first plugin outermost.
- **Rust:** The same chain, and plugin i's wrap brackets each poll and drop of the user function. Plugin i's `on_user_function_start` and `on_user_function_end` run inside plugin i-1's `wrap_user_function`, before and after plugin i's wrap.
- **Why:** A Rust future runs in many polls on any thread. Bracketing each poll is what `tracing::Instrumented`, OpenTelemetry's `WithContext` and tokio's `LocalKey::scope` do.
- **Decision:** Same capability in Rust form.
- **Where:** `src/plugin.rs`, `Plugin::wrap_user_function`; `src/plugin/scope.rs`, `run_user_function` and `OwnChain::level`.

### Skipping or replacing user code

- **JavaScript:** A wrapper that returns without calling `fn` skips it, and the wrapper's return value is recorded.
- **Rust:** The SDK runs the user function exactly once and records its value, and `wrap_invocation` cannot replace the invocation's output. If a wrap returns without calling `next()`, the SDK calls it.
- **Why:** The recorded value is what replay returns, so a plugin could change a durable result.
- **Decision:** Decision (2026-11-04): a plugin cannot skip a user function or replace its result, as in Python.
- **Where:** `src/plugin/scope.rs`, `run_user_function`.

### Driving user code

- **JavaScript:** `fn` is a value; a wrapper may call it later, elsewhere, or not at all.
- **Rust:** `next` is a borrowed, non-`Send` `&mut dyn FnMut()`, valid only during the wrap call and not a future.
- **Why:** With the body's driver in plugin code, every guarantee about the body becomes a protocol between threads. A borrowed closure makes each guarantee a property of ownership.
- **Decision:** Rust language.
- **Where:** `src/plugin.rs`, `Plugin::wrap_user_function`.

### Context on tasks

- **JavaScript:** Async context follows every promise, including work user code starts.
- **Rust:** Each task the SDK starts for an operation (`.spawn()`, map items, parallel branches, combinator inputs) runs inside the wraps and `tracing` span current when the builder became its future (`.await`, `.spawn()` or `.future()`), or when the container dispatched the item or branch; the handler's root task runs inside the invocation's wraps, captured at its dispatch. A user `tokio::spawn` and a hook future need an explicit carrier: tracing (`.instrument(Span::current())`), direct OpenTelemetry (capture `Context::current()`, then `.with_context(captured)`), or the plugin's own mechanism.
- **Why:** Rust has no implicit async context, and the SDK does not start a user's task.
- **Decision:** Same capability in Rust form.
- **Where:** `src/task.rs`, `TaskContext`, `spawn_tracked` and `spawn_root`; `src/future.rs`, `DurableFuture::for_operation`, `DurableFuture::for_started_operation` and `DurableFuture::for_start_step`; `src/map_parallel.rs`, `MapDispatch::spawn_item` and `BranchDispatch::spawn_item`.

### Panicking user function

- **JavaScript:** A thrown error is an outcome the wrapper sees.
- **Rust:** An `Err` reaches `on_user_function_end`. A panic fails the invocation, and no end hook fires.
- **Why:** Calling plugin code during the unwind would put plugin code between the panic and the task gateway, which resumes the original payload.
- **Decision:** Rust language.
- **Where:** `src/plugin/scope.rs`, `run_user_function`.

### Value in the end hook

- **JavaScript:** A wrapper observes the exact value or rejection of `fn`.
- **Rust:** `on_user_function_end` receives `Result<(), &WireError>`, `Ok(())` or the `WireError` the SDK records, never the value. `on_operation_end` can read the serialized checkpoint payload through `OperationInfo::result()`, which is `None` when the terminal update carries no payload.
- **Why:** `Plugin` is used as `dyn Plugin`, so no method can be generic over an operation's output type. A hook future is `'static`, so it cannot borrow the value.
- **Decision:** Decision (2026-11-04): the end hook receives `Result<(), &WireError>`, never the value, as in Python.
- **Where:** `src/plugin.rs`, `Plugin::on_user_function_end` and `OperationInfo::result`.

### Hook errors

- **JavaScript:** Swallows hook and wrapper failures and logs nothing.
- **Rust:** Contains the failure and logs one WARN on target `aws_durable_execution_sdk::plugin` with `plugin` = `<plugin name> (plugins[i])` (`plugins[i]` when `name()` panicked) and `hook` = the method name. Nothing else changes, except that a plugin whose synchronous start hook failed gets no wrap and no end hook for that user function, and a panic in the destructor of a returned error fails the invocation.
- **Why:** A broken plugin otherwise gives no signal, and the outcome stays identical.
- **Decision:** Decision (2026-11-04): contain hook and wrap failures and log them at WARN.
- **Where:** `src/plugin.rs`, `InvocationPlugins::dispatch_to`, `report_hook_error` and `log_hook_panic`; `src/plugin/scope.rs`, `OwnChain::level`.

### Scope close and end hooks

- **JavaScript:** A child wrapper returns after `fn` settles, and JavaScript accepts the descendants' checkpoints until then.
- **Rust:** A scope closes when its raw body returns, before the SDK derives its `WireError` and calls the end hooks, and stops its tracked tasks, so a stopped descendant gets no end hook. The end hook's synchronous part runs in the poll in which the body returned, so a stop cannot interrupt it.
- **Why:** Work that outlives its scope has no owner that records it.
- **Decision:** Decision (2026-11-03): Rust stops the `.future()` inputs a combinator started, the work inside a child context or batch that has closed, and all tracked operation work when the drive ends.
- **Where:** `src/plugin/scope.rs`, `run_user_function`; `src/child.rs`, `ChildExecution::run`; `src/checkpoint_coalescer.rs`, `ScopeClose`.

### `with_retry` block

- **JavaScript:** With its default `wrapWithRunInChildContext: true`, `withRetry` calls each attempt inside one outer child.
- **Rust:** The outer `with_retry` block is a user-function site. Its `attempt-N` child contexts have no own frame and no user-function hooks, run inside the outer block's wraps, and keep their operation hooks.
- **Why:** The block is one outer site rather than one site per attempt, as in JavaScript.
- **Decision:** Same capability in Rust form.
- **Where:** `src/builders/with_retry.rs`, `WithRetryBuilder::build_child`; `src/with_retry.rs`, `start_attempt`.

### `with_retry` without the outer child

- **JavaScript:** `wrapWithRunInChildContext: false` runs the attempts in the caller's context, with no wrapping child.
- **Rust:** `WithRetryBuilder::without_child_context()` runs the `attempt-N` child contexts and the retry waits in the caller's context, with no block record. The block is no user-function site, and its `attempt-N` children fire no user-function hooks.
- **Why:** The option groups or ungroups the attempts in the history, as in JavaScript. The row "`with_retry` structure" gives the history shape.
- **Decision:** Same capability in Rust form.
- **Where:** `src/builders/with_retry.rs`, `WithRetryBuilder::without_child_context` and `WithRetryBuilder::build_unwrapped`.

### Asynchronous plugin work around user code

- **JavaScript:** A wrapper may await before calling `fn` and after `fn` settles, and the operation waits for it.
- **Rust:** A start or end hook future runs on its own task, and the operation never waits for it. The invocation-end drain waits for it until 500 ms before the deadline.
- **Why:** JavaScript waits for a wrapper only because the wrapper calls the user function. In Rust the SDK calls it, so JavaScript's rule that the SDK never waits for an operation plugin hook applies.
- **Decision:** Decision (2026-11-04): plugin work around user code is fire and forget.
- **Where:** `src/plugin.rs`, `Plugin::on_user_function_start`, `Plugin::on_user_function_end` and `InvocationPlugins::start_work`.

### Context of an inline await

- **JavaScript:** An operation's wrapper runs in the async context of the code that created the operation.
- **Rust:** An operation awaited directly runs inside the wraps of the code that polls it, plus its own frame, so one created in context A and awaited in B sees B, never A. An operation on its own task runs in the context current when it became its future, and a lazy step input that a combinator polled in B begins nests as `[I, b, I, a]`, with its captured A innermost.
- **Why:** Rust cannot leave the poller's stack frame while it polls, so a directly awaited operation's OpenTelemetry and `tracing` parentage follows the poller, where JavaScript follows the creator.
- **Decision:** Decision (2026-11-03): an operation starts when something first polls it.
- **Where:** `src/plugin/scope.rs`, `Frames::enter`; `src/future.rs`, `DurableFuture::begin_now` and `DurableFuture::begin_in_creator_context`.

### Abandoned hook work

- **JavaScript:** Leaves hook work that has not settled pending at the drain deadline.
- **Rust:** Aborts and joins it before `on_invocation_end`.
- **Why:** A pending JavaScript promise can resume after a thaw against a plugin instance whose invocation ended. That breaks the one-instance-per-invocation rule.
- **Decision:** JavaScript defect and Decision (2026-10-31): abort and join hook work at the drain deadline.
- **Where:** `src/plugin.rs`, `InvocationPlugins::invocation_end`.

### Abandoned item end hook

- **JavaScript:** A `map` item or `parallel` branch abandoned at early completion, whose body returns without awaiting a dropped checkpoint, fires `onOperationEnd` with `SUCCEEDED` and `isReplay` false. The service never records that `SUCCEED`, and the batch result does not include the item.
- **Rust:** Fires no end hook for an abandoned item or for its unfinished inner operations.
- **Why:** The JavaScript hook reports a success the history never records and the batch never used. Rust drops the abandoned item's future, so its body cannot return.
- **Decision:** JavaScript defect.
- **Where:** `src/map_parallel.rs`, `produce_batch`.

### Trace header

- **JavaScript:** `providePropagationMetadata` returns a metadata object.
- **Rust:** `propagation_trace_id` returns `Result<Option<String>, BoxError>`.
- **Why:** Only the header value is used, and a later field is a new trait method with a default body.
- **Decision:** Same capability in Rust form.
- **Where:** `src/plugin.rs`, `Plugin::propagation_trace_id`.

### Child operations depth

- **JavaScript:** Rejects negative, fractional and NaN values at startup.
- **Rust:** A `u32`, so every value is valid.
- **Why:** The type rules out the invalid values.
- **Decision:** Same capability in Rust form.
- **Where:** `src/options.rs`, `OptionsBuilder::plugin_child_operations_depth`.

### Combinator records within the depth

- **JavaScript:** A combinator runs as a child context, so within `childOperationsDepth` its record carries `ReplayChildren`, as every child context's does, and replay runs the combinator again over its inputs.
- **Rust:** Within the depth, a `try_join_all` or `join_all` `SUCCEED` and every combinator `FAIL` carry `ReplayChildren`, as in JavaScript, and replay of a flagged `try_join_all` or `join_all` drives its inputs again. A `race` or `select_ok` `SUCCEED` carries the flag only above the size limit, and replay returns the recorded winner.
- **Why:** For `race` and `select_ok`, a flagged record tells replay to drive only the recorded winner. Running the race again, as JavaScript does, can return a different winner.
- **Decision:** JavaScript defect and Decision (2026-10-31): flag every combinator record within the depth except a `race` or `select_ok` `SUCCEED`.
- **Where:** `src/combinator.rs`, `checkpoint_collection_succeed`, `checkpoint_winner_succeed` and `checkpoint_fail`.

### Virtual child hooks

- **JavaScript:** The start and end hooks of a virtual child report `isReplay: true`, even on the first run.
- **Rust:** Each hook reports the invocation's replay state when it fires, and `OperationInfo::is_virtual` is `true` only for a virtual child. Flat map items and parallel branches are virtual children.
- **Why:** In JavaScript a plugin that skips replayed starts never traces a virtual child, because its first run is reported as a replay.
- **Decision:** JavaScript defect and Decision (2026-10-28): report the invocation's replay state.
- **Where:** `src/child.rs`, `ChildExecution::execute_virtual`; `src/plugin.rs`, `OperationInfo::is_virtual`.

### Operation map

- **JavaScript:** Includes the `EXECUTION` operation, in the order the service returned the operations.
- **Rust:** Includes the `EXECUTION` operation in a `BTreeMap` ordered by wire ID.
- **Why:** The checkpoint log is a hash map and keeps no service order. A sorted map gives every plugin the same order on every invocation.
- **Decision:** Decision (2026-10-31): order the map by wire ID.
- **Where:** `src/plugin.rs`, `InvocationInfo::operations`.

### Replayed CANCELLED invoke

- **JavaScript:** Fires a start hook and no end hook.
- **Rust:** Fires the end hook, as for every terminal status.
- **Why:** `CANCELLED` is terminal, and JavaScript's replay misses it, so a plugin opens a span it never closes.
- **Decision:** JavaScript defect.
- **Where:** `src/invoke.rs`, `recorded_outcome`.

### Attempt-end timestamps

- **JavaScript:** A retried step attempt's end info has no end timestamp, and a condition check's has its start timestamp set to the end time.
- **Rust:** Every attempt-end info has an end timestamp, the record's or the time the checkpoint call returned. A condition check keeps its start timestamp.
- **Why:** A plugin needs an attempt's end time to close a span. Python sets it the same way, and conformance requirement 10-21 asserts it.
- **Decision:** JavaScript defect and Decision (2026-10-28): give every attempt-end info an end timestamp.
- **Where:** `src/plugin.rs`, `attempt_ended` and `with_observed_end`, which `src/step.rs` and `src/wait_for_condition.rs` call when an attempt ends.

## Testing

### `LocalRunner` scope

- **JavaScript:** A local runner with `skipTime`, setup, teardown and reset, a live waiter, and local timers that time out a callback (`Callback timed out`) and a missed heartbeat.
- **Rust:** Always skips time on a virtual clock, needs no setup, teardown or reset, adds fault injection and `max_invocations`, and lets a test react to a run in progress through `RunHandle` and `OperationHandle` (`wait_for_data`, `send_callback`, `send_callback_heartbeat`, `pause`, `resume`). It runs no callback, heartbeat or execution timeout timers: `CallbackReply::timeout()` and `CallbackReply::heartbeat_timeout()` deliver those records, and `send_callback_heartbeat` validates and starts nothing.
- **Why:** It is a test framework, not a service emulator; JavaScript's local callback timers emulate the service on the checkpoint server's real clock even with `skipTime`. A scripted reply tests the same handler branch at once.
- **Decision:** Decision (2026-10-31): run no service timers, and deliver timeout records through scripted replies.
- **Where:** `src/test_util.rs`, `LocalRunner`; `src/test_util/live.rs`, `RunHandle` and `OperationHandle`.

### Backend validation

- **JavaScript:** Validates the full update model.
- **Rust:** Checks the service size and delay limits, registered chained-invoke targets, known operation types, an existing operation's identity, each type's actions and status transitions, CONTEXT parents and repeated IDs, with JavaScript's rules and messages. It does not check the request model, ID and name patterns, each type's option and payload rules, or the count and position of EXECUTION updates.
- **Why:** The added checks catch a corrupt history without making `LocalRunner` a full model of the service. The remaining service-model checks stay out of scope.
- **Decision:** Decision (2026-11-03): add the history checks, and keep the request-shape checks out of scope.
- **Where:** `src/test_util.rs`, `Backend::checkpoint_counted`, `validate_service_limits` and `validate_history_rules`.

### Unregistered invoke target

- **JavaScript:** Throws `No function found for function name <name>`, which rejects the whole run.
- **Rust:** Rejects the START checkpoint, so the run fails with a message that names the function.
- **Why:** The test fails in both, with a message naming the function.
- **Decision:** Same capability in Rust form.
- **Where:** `src/test_util.rs`, `Backend::checkpoint_counted`.

### Invoke target that returns nothing

- **JavaScript:** Records the invoke `FAILED` with no error.
- **Rust:** Records it `SUCCEEDED` with result `null`.
- **Why:** In JavaScript a plain function that succeeds with no return value looks failed and carries no error.
- **Decision:** JavaScript defect.
- **Where:** `src/test_util.rs`, `LocalRunner::register_function`.

### Nested durable run that ends suspended

- **JavaScript:** Awaits the nested run and throws if it returns `RUNNING`. A target that waits on an unanswered callback never returns, so the test waits until the test framework's timeout.
- **Rust:** Leaves the caller's invoke started, and the caller's run ends suspended, naming the pending invoke.
- **Why:** The service keeps the caller's invoke pending in that case, and JavaScript holds the test until the test framework's timeout instead.
- **Decision:** JavaScript defect.
- **Where:** `src/test_util.rs`, `LocalRunner::drive_nested`, `Backend::record_invoke_outcome` and `suspended_message`.
