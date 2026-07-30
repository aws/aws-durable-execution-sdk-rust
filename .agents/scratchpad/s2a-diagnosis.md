# S2a diagnosis — spawn suspension scoping and the four failing callback examples

Status: COMPLETE.

## Verdict in one line

The two symptoms do **not** share a cause. Symptom 1 (`.spawn()` parks the
ROOT scope) is a real SDK defect in `src/future.rs` / `src/builders.rs`.
Symptom 2 (four callback examples end TIMED_OUT/FAILED in the Cloud tests
workflow) is a **test-harness** defect in `examples/cloud/run_cloud_tests.sh`;
the SDK's un-timed callback suspension is recorded correctly and resumes
correctly. The leading hypothesis ("an un-timed callback wait parks the
invocation without recording a resumable suspension") is **REFUTED**.

## Symptom 2 — refuted, with evidence

### Evidence 1: the CI run at aa6e365 (read-only, account 352587389722)

GitHub Actions run 30516632123 ("Cloud tests", alpha, aa6e365). Immediately
before every one of the four "completed callback" lines the AWS CLI reports a
client-side rejection:

```
2026-07-30T05:43:50.6629200Z aws: [ERROR]: Invalid base64: ""approved by cloud test""
2026-07-30T05:43:50.7329586Z 05:43:50 create_callback_simple: completed callback
2026-07-30T05:46:50.7812888Z aws: [ERROR]: Invalid base64: ""approved by cloud test""
2026-07-30T05:46:50.8505504Z 05:46:50 create_callback_heartbeat: completed callback
2026-07-30T05:47:52.6864569Z aws: [ERROR]: Invalid base64: ""approved by cloud test""
2026-07-30T05:47:52.7573922Z 05:47:52 wait_for_callback_basic: completed callback
2026-07-30T05:50:52.6570545Z aws: [ERROR]: Invalid base64: ""approved by cloud test""
2026-07-30T05:50:52.7309859Z 05:50:52 wait_for_callback_submitter: completed callback
```

Those are exactly the four failures, and exactly the four rows in the
EXPECTATIONS table carrying `drive=callback`
(`examples/cloud/run_cloud_tests.sh:106,107,121,122`). No other example in the
46 asks the harness to complete a callback. The two "passing" callback
examples (`create_callback_timeout`, `wait_for_callback_timeout`) are NOT
driven by the harness at all — they are expected to time out and catch it, so
they never depend on a callback completion.

`drive_callback` (`examples/cloud/run_cloud_tests.sh:176-189`) sends:

```sh
aws lambda send-durable-execution-callback-success \
    --callback-id "$cb_id" --result "$CALLBACK_RESULT" > /dev/null
log "$1: completed callback"
```

`--result` is a **blob** parameter. AWS CLI v2 defaults `cli_binary_format` to
`base64`, so the CLI tries to base64-DECODE `"approved by cloud test"`, fails,
and never calls the API. The exit status is never checked (stdout goes to
`/dev/null`, and `set -e` does not apply because the function is invoked in an
`if ! drive_callback ...` condition), so the harness logs "completed callback"
after a send that never happened. The execution then sits parked until the
`DurableConfig.ExecutionTimeout: 300` fires — matching the observed "about six
minutes after invoke" (start + 300 s + poll latency).

`create_callback_heartbeat` ends FAILED rather than TIMED_OUT for the same
reason: with no completion and no heartbeat traffic, its 10 s heartbeat
deadline expires, the service records a callback failure, the execution
resumes, and `cb.result().await?` propagates the error — correct SDK
behaviour for an abandoned callback.

### Evidence 2: narrow cloud repro (owner account 468733773112, us-west-2)

Deployed ONLY `examples/template_external.yaml` as stack `s2a-diag-external`,
with `CreateCallbackSimple`'s `bootstrap` rebuilt from the alpha working tree
at aa6e365 (so the un-timed path sends NO `CallbackOptions`, as alpha does).
Invoked `create_callback_simple` once, execution
`.../durable-execution/e725aa27-2d48-430a-aead-1fefde4d6832/3fff9a48-e331-34af-a7e2-9b4490a9446a`.

History while parked — a suspension **is** recorded and the execution is
RUNNING, not stalled in some unrecorded state:

```
ExecutionStarted      {"Input": {"Truncated": true}, "ExecutionTimeout": 300}
CallbackStarted       {"CallbackId": "Ab9hZXjGYXJuOmF3czpsYW1iZGE6..."}
InvocationCompleted   {"StartTimestamp": "2026-07-30T18:07:49.871Z",
                       "EndTimestamp": "2026-07-30T18:07:51.669Z", ...}
status: RUNNING
```

Experiment A — the harness's exact command form:

```
$ aws lambda send-durable-execution-callback-success \
      --callback-id "$CB" --result '"approved by cloud test"'
Invalid base64: ""approved by cloud test""
exit=255
```

Experiment B — same callback id, same result, one extra flag:

```
$ aws lambda send-durable-execution-callback-success \
      --callback-id "$CB" --result '"approved by cloud test"' \
      --cli-binary-format raw-in-base64-out
exit=0
```

History 12 s later:

```
CallbackStarted
InvocationCompleted  18:07:49.871 -> 18:07:51.669   (first invocation, PENDING)
CallbackSucceeded    {"Result": {"Truncated": true}}
InvocationCompleted  18:08:17.813 -> 18:08:17.848   (SECOND invocation = resume)
ExecutionSucceeded   {"Result": {"Truncated": true}}
```

and the still-open `lambda invoke` returned `"approved by cloud test"`.

So for an un-timed callback the SDK: records `CallbackStarted`, returns
`{"Status":"PENDING"}`, the service parks the execution, the external
completion wakes it, a **second invocation** occurs 26 s later, replay reads
the settled record and the execution SUCCEEDS. Nothing about the un-timed
path is unresumable, and the absence of `CallbackOptions` on the wire
(`src/callback.rs:146-149`, `build_callback_options` returning `None` when
both timeout and heartbeat are zero) does not affect resumption.

### Fix for symptom 2 (a later slice; NOT an SDK change)

`examples/cloud/run_cloud_tests.sh` `drive_callback`: add
`--cli-binary-format raw-in-base64-out` to the send, and check its exit
status so a failed send fails the example instead of being logged as
"completed callback". No `src/` change is warranted. Note the same class of
bug is already avoided for `aws lambda invoke --payload` in the same script,
which does pass the flag.

## Symptom 1 — real, and independent

### Proven defect

`.spawn()` gives the spawned operation NO suspension scope of its own. Every
builder's `spawn()` does the same three lines (`src/builders.rs:193-201` and
twelve siblings) and hands the operation future to
`DurableFuture::spawn_blessed` (`src/future.rs:67-131`), which `tokio::spawn`s
it and drives it directly — no `new_child_scope`, no `drive_scope`. The
operation therefore runs on the SPAWNING context's scope, which for a
top-level `.spawn()` is the ROOT. Any parking operation inside it calls
`ctx.suspend_now()` (`src/context.rs:339-342`) → `SuspensionSignal::request_suspend`
(`src/driver.rs:134-144`) on the ROOT signal. `drive_invocation`
(`src/driver.rs:334-402`) returns `Poll::Ready(InvocationOutcome::Pending)` on
the next poll, drops the handler future, and dropping it drops every sibling's
`AbortOnDrop` guard (`src/driver.rs:306-322`, held inside the returned
`DurableFuture`), aborting sibling spawned tasks mid-flight.

Observed, from the three new `#[ignore]`d tests in
`src/driver.rs` (`mod spawn_scope_regressions`), run with `--ignored`:

```
running 3 tests
thread '...::spawned_step_reaches_terminal_checkpoint_before_pending' panicked at src/driver.rs:
the spawned step body must run to completion before the invocation suspends; a parked spawned sibling must not abort it
thread '...::joined_spawned_wait_and_step_lets_the_step_finish' panicked at src/driver.rs:
join! over a spawned wait and a spawned step must not abort the step
thread '...::spawned_wait_parks_then_owner_does_sequential_step' panicked at src/driver.rs:
the non-spawned step must complete before Pending; a parked spawn must not pre-empt the owner
test result: FAILED. 0 passed; 3 failed; 0 ignored; ... filtered out
```

All three FAIL (they do not hang — each is wrapped in a 5 s
`tokio::time::timeout`, and the whole run finished in < 1 s). The assertion
that fires is the one checking that the step body ran to completion BEFORE the
invocation reported Pending, confirming that today's code prematurely aborts
runnable work.

The existing test
`src/driver.rs:1175 suspend_drops_active_sibling_and_makes_no_further_checkpoint`
asserts today's behaviour ("sibling task must be aborted on suspend") and
therefore must be revised by the fix slice; it is the only test that pins the
defect as intended behaviour.

### The two symptoms do NOT share a cause

Symptom 2 is entirely `examples/cloud/run_cloud_tests.sh` (evidence above).
None of the four failing examples uses `.spawn()` at all — each handler is a
single sequential `create_callback`/`wait_for_callback` await
(`examples/external/create_callback_simple/src/main.rs:19-25` and the three
peers), so no spawned scope is involved, and the cloud repro shows that path
suspending and resuming correctly. Conversely symptom 1 reproduces in-process
with no service involvement. One defect in `src/`, one defect in the harness.

### Sites that must change (fix slice)

- `src/future.rs:67-131` — `spawn_blessed`: take a child scope, drive the
  operation with `drive_scope`, and record the spawn's settle transition ON
  THE SPAWNED TASK.
- `src/builders.rs:200, 317, 500, 667, 862, 1002, 1182, 1458, 1663, 1777, 1883,
  1986, 2089` — the thirteen `spawn()` terminals. They must mint the child
  scope, rebind the builder's context onto it, and register the spawn. Extract
  one helper (fn or macro) rather than editing thirteen copies.
- `src/context.rs` — a `with_spawn_scope`-style rebind plus the per-scope
  `ScopeQuiescence` field, alongside the existing `new_scoped_child` /
  `new_scoped_flat_child` (`src/context.rs:223,246`); `suspend_now`
  (`src/context.rs:339`) must mediate through the quiescence tracker.
- `src/driver.rs:74-165` (`SuspensionSignal`) — own the per-scope quiescence
  state; `src/driver.rs:334-402` (`drive_invocation`) — gate the "report
  suspension" decision on scope quiescence; also intercept handler completion
  with parked spawns.
- `src/driver.rs:1175` — revise
  `suspend_drops_active_sibling_and_makes_no_further_checkpoint` for the
  straggler-timeout semantics;
  `src/driver.rs` `mod spawn_scope_regressions` — drop the three `#[ignore]`s.
- `src/future.rs:20-30` (the `DurableFuture` doc block) and each `spawn()`
  rustdoc — state the new guarantee.

## Intended mechanism — precise specification

### Shared primitive: scope isolation + drive_scope

Both map_parallel branches and spawned tasks share **one mechanism** for scope
isolation:

1. `SuspensionSignal::new_child_scope()` (`src/driver.rs:119`) — creates a
   child scope with its own suspension flag, sharing the invocation waker.
2. `drive_scope(future, scope)` (`src/driver.rs:431-470`) — polls the future
   until completion or until the child scope's flag is set, returning
   `ScopeOutcome::Completed(T)` or `ScopeOutcome::Suspended`.

Map_parallel: the coordinator calls `new_scoped_child()`/`new_scoped_flat_child()`
on the context, which internally calls `new_child_scope()`, and
`execute_single_item` calls `drive_scope(run_item(child_ctx, index), scope)`.
The coordinator's own loop is the quiescence evaluator — it checks
`any_suspended && !stopped && running == 0` and calls `ctx.suspend_now()`.

Spawns: `spawn_blessed` calls `new_child_scope()` on the owner's scope, and
the spawned tokio task calls `drive_scope(operation_future, child_scope)`. The
quiescence evaluator is distributed across the task, the handle, and the
driver (see below).

### Quiescence state: `ScopeQuiescence`

One instance per `SuspensionSignal`. Created when the scope is created.
Sequential children (via `new_child`) share their parent's scope and therefore
its quiescence state. Branch scopes and spawn scopes each get their own (but
only the OWNER's scope tracks spawns — the child scope has its own empty
quiescence).

```rust
/// Tracks the settle state of spawned tasks under this scope.
/// Lives on SuspensionSignal; one per scope.
pub(crate) struct ScopeQuiescence {
    /// Number of spawned tasks that have not yet settled (completed or
    /// parked). Incremented BEFORE tokio::spawn, decremented by the task
    /// AFTER drive_scope returns.
    spawns_outstanding: AtomicUsize,
    /// True if at least one spawned task settled as parked (its child scope
    /// was suspended). Once set, never cleared within an invocation.
    any_spawn_parked: AtomicBool,
    /// True if the owner called suspend_now() while spawns were still
    /// outstanding. Cleared when quiescence fires (or never set if the
    /// owner completes normally).
    owner_parked: AtomicBool,
}
```

### State transitions (exhaustive)

**Increment (owner thread, before tokio::spawn):**
```
spawns_outstanding += 1
```
Done synchronously before `tokio::spawn` so the count is correct even if the
task resolves before the owner resumes.

**Task settles — completed (spawned task, after drive_scope returns Completed):**
```
spawns_outstanding -= 1
if spawns_outstanding == 0 && owner_parked {
    scope.request_suspend()   // fires quiescence
}
```
The task then sends `SpawnResult::Completed(value)` on the oneshot.

**Task settles — parked (spawned task, after drive_scope returns Suspended):**
```
spawns_outstanding -= 1
any_spawn_parked = true
if spawns_outstanding == 0 && owner_parked {
    scope.request_suspend()   // fires quiescence
}
```
The task then sends `SpawnResult::Parked` on the oneshot. It does NOT call
`request_suspend()` on the OWNER's scope unconditionally — only when
`owner_parked` is true and it's the last outstanding.

**Task aborted (RAII guard inside the tokio::spawn body, on drop):**
```
if !settled {
    spawns_outstanding -= 1
    // No quiescence check: an abort means the scope is being torn down.
}
```
The guard is set to `settled = true` just before the task sends on the
oneshot, so it fires only for tasks cancelled by AbortOnDrop (dropped handle)
or runtime shutdown.

**Owner parks — suspend_now() (owner thread):**
```
if spawns_outstanding == 0 {
    scope.request_suspend()   // immediate: all spawns settled
} else {
    owner_parked = true       // deferred: wait for spawns to settle
}
```
Then awaits `pending::<T>()` as today. The driver observes `request_suspend()`
on the next poll.

**Handle receives SpawnResult::Parked (polled by the owner):**
```
scope.request_suspend()       // the owner is blocked on this handle
// then: std::future::pending::<T>().await (handle never resolves)
```
This is the trigger for the `tokio::join!` pattern: when the owner polls a
parked handle, the scope is flagged. If the owner never polls the handle,
this path never fires — quiescence comes from the driver-side check instead.

**Driver intercepts handler completion with parked spawns:**

In `drive_invocation`, when the handler future returns
`Poll::Ready(Ok(result))` or `Poll::Ready(Err(err))`:
```
if scope_quiescence.any_spawn_parked() {
    return Poll::Ready(InvocationOutcome::Pending)
}
```
This handles the case where the owner completes without ever awaiting the
parked handle (e.g., `let _wait = ctx.wait(10s).spawn(); Ok("done")`). The
parked spawn's durable state is already recorded; returning Pending ensures
the service re-invokes to resume it.

### Correctness proof for all cases

**Case A: `wait.spawn(); step.spawn(); tokio::join!(wait, step)`**
1. spawns_outstanding = 2
2. wait task parks → outstanding=1, any_spawn_parked=true
3. step task completes → outstanding=0. Checks: owner_parked? No. No flag.
4. step sends Completed(Ok(7)) on oneshot.
5. tokio::join! polls both handles. Step → Ready(Ok(7)). Wait → receives
   Parked, calls request_suspend(), awaits pending() → Pending.
6. tokio::join! yields Pending. Handler yields Pending.
7. Driver: is_suspend_requested() → TRUE. Returns Pending. ✓
8. Step's Succeed checkpoint was recorded before (3). ✓

**Case B: `let _wait = wait.spawn(); step(work).await; Ok("done")`**
1. spawns_outstanding = 1
2. wait task parks → outstanding=0, any_spawn_parked=true. owner_parked? No.
3. Sequential step runs normally, checkpoints Succeed.
4. Handler returns Ok("done").
5. Driver: Poll::Ready(Ok("done")). Checks any_spawn_parked → TRUE. Returns
   Pending. ✓
6. The step completed fully before (4). ✓

**Case C: `wait.spawn()` only, handler returns immediately.**
Same as Case B but without the sequential step: handler returns Ok, driver
sees any_spawn_parked, returns Pending. ✓

**Case D: `wait.await` (owner explicitly parks via suspend_now)**
1. spawns_outstanding = 0 (no spawns at all)
2. suspend_now(): outstanding==0 → request_suspend() immediately.
3. Handler at pending(). Driver: is_suspend_requested() → TRUE. Returns
   Pending. ✓

**Case E: `wait.spawn(); expensive_step.spawn(); join!(wait, step)` with
step slower than wait**
Same as Case A — the step checkpoint lands before (3) because drive_scope
doesn't drop the step on the wait's park (child scope isolation). ✓

**Case F: straggler — spawned step body does `std::future::pending()`**
1. spawns_outstanding = 1
2. The step task never returns from drive_scope (the body never completes, and
   no durable operation calls request_suspend on the child scope).
3. If the owner also calls suspend_now(): owner_parked=true, but
   outstanding=1, so quiescence defers. The invocation is stuck until the
   Lambda timeout fires, which drops the handler future and its AbortOnDrop
   guard, aborting the task.
4. This is CORRECT and BOUNDED: a non-durable pending body is a user bug (no
   durable state to resume from), and the Lambda timeout is the backstop.
   Document in the `spawn()` rustdoc: a spawned body must either complete or
   reach a durable suspension point; bodies that block on non-durable futures
   hold the invocation until the Lambda timeout and are aborted.

**Case G: the existing test at driver.rs:1175 (spawned step body hangs)**
This is Case F. The fix slice changes the test's assertion: the step is still
aborted, but by the timeout/drop path, not by a premature scope flag. The
test validates that no checkpoint is made after the abort. The test must add
a bounded tokio::time::timeout wrapper.

### How this avoids the `failed/1478095` deadlock

That attempt had the right scope isolation but put BOTH the accounting
transitions (mark_completed, mark_parked) AND the quiescence evaluation in
the RETURNED HANDLE's async body (`failed/1478095:src/future.rs:148-190`).
The transitions fire only after `rx.await` inside the future returned by
`spawn_blessed`, so a handle the user never polls never advances the counter.
In both of its hanging tests the handler does `let _sib = ...spawn();` and
then awaits something that never returns, so `_sib` is never polled:
`spawns_running` stays 1, `request_owner_park()` defers forever, the handler
future has no waker to fire, `drive_invocation` returns `Poll::Pending`
forever, and `cargo test` never terminates. Its `SpawnAccountingGuard`
(`failed/1478095:src/future.rs:200-217`) cannot break the cycle because the
guard is dropped only when the driver drops the handler — i.e., only after the
suspension it is blocking.

Additionally, its docs and code disagreed on tracker granularity:
`failed/1478095 src/builders.rs:20-30` called it "the invocation-wide
quiescence tracker" while `src/context.rs:250,278,331` constructed a fresh
`SpawnQuiescence::new(scope)` per scope.

Four rules follow, all of which the fix must keep:

1. **Accounting advances on the tokio task, never on a handle.** The runtime
   always polls a spawned task; nothing guarantees a handle is polled.
   `mark_completed()` and `mark_parked()` are called by the task body, not
   the returned future.

2. **The handle propagates, it does not account.** The handle's only role is
   calling `request_suspend()` when the owner polls a parked handle. If the
   handle is never polled, the driver's completion-with-parked-spawns check
   provides the same outcome.

3. **The driver decides when the outcome changes.** `drive_invocation` owns
   the `Poll::Ready(Pending)` transition. The quiescence state on the scope
   only SETS the `request_suspend` flag; the driver reads it and acts.

4. **Quiescence must be reachable in bounded time.** Every spawn must settle:
   completed, durably parked, or aborted. A spawn whose body blocks on a
   non-durable future never settles, and the Lambda timeout is the backstop
   (it drops the handler, which drops AbortOnDrop, which aborts the task).
   This is documented as a contract on `spawn()` bodies.

### Straggler policy (bounded non-durable hang)

A "straggler" is a spawn whose body blocks on a non-durable future
indefinitely: it never calls `request_suspend()` on its child scope, and
`drive_scope` never returns.

Policy: **no special deadline mechanism**. The Lambda platform's execution
timeout (configurable, default 900s for durable, set via
`DurableConfig.ExecutionTimeout`) is the backstop. When the timeout fires,
the Lambda runtime drops the handler future. Dropping the handler drops
`AbortOnDrop` guards, which abort straggler tasks. The invocation ends as
TIMED_OUT, which is correct: no durable suspension was recorded for the
straggler, so there is nothing to resume from.

This is strictly better than today's behavior: today the straggler is aborted
IMMEDIATELY (the wait's ROOT-scope park fires), which also means "nothing to
resume from", but ALSO aborts siblings that HAVE durable state and forces them
to re-execute. With the fix, durable siblings complete normally, and only the
non-durable straggler awaits the timeout.

The `spawn()` rustdoc must document: a spawned body that blocks on
non-durable futures holds the invocation until the Lambda timeout. Bodies
should reach either completion or a durable suspension point.

The existing test at `src/driver.rs:1175` is adapted by the fix slice to
validate this: the straggler is aborted when the handler is dropped (after a
bounded timeout in the test), no further checkpoints are made, and durable
siblings that completed before the straggler have their checkpoints preserved.

### Advertised semantics the fix would alter

- `DurableFuture`'s doc (`src/future.rs:20-30`) and every `spawn()` rustdoc
  gain a guarantee they do not make today: a parked spawned operation does not
  abort runnable spawned siblings, and the invocation suspends only at
  quiescence.
- A spawned operation that parks now yields a handle that never resolves in
  this invocation (identical to a parked map/parallel branch) instead of
  taking the whole invocation down with it. Code shaped like
  `let (_, r) = tokio::join!(wait, work);` still never returns on the parking
  invocation — the difference is that `work`'s side effect is checkpointed
  before the drop, so it is not re-executed on resume.
- When the handler completes (returns Ok/Err) while a spawn is durably parked,
  the invocation reports PENDING instead of the handler's result. This is new:
  today such a pattern does not arise because the park fires immediately.
- Abort-on-drop is unchanged: dropping a handle still cancels its task.
- Suspension stays unswallowable: the flag is still set from inside the
  engine, and the driver still drops the handler future rather than resuming it.
- No public API surface changes; `ScopeQuiescence` and the scope plumbing are
  `pub(crate)`.
- A non-durable straggler now holds the invocation until the Lambda timeout
  instead of being aborted immediately. Documented as a contract on spawn
  bodies.

## Reproduction / teardown notes

- Cloud repro used stack `s2a-diag-external` in 468733773112 (us-west-2) and
  an IAM role `lambda-execution` created for it. **TEARDOWN STATUS: BLOCKED BY
  OPERATOR.** Three attempts to delete via `use_aws` tool were denied by the
  user's tool authorization gate. The resources remain and require manual
  deletion:
  ```
  aws cloudformation delete-stack --stack-name s2a-diag-external --region us-west-2
  aws cloudformation wait stack-delete-complete --stack-name s2a-diag-external --region us-west-2
  aws iam detach-role-policy --role-name lambda-execution \
      --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  aws iam delete-role-policy --role-name lambda-execution --policy-name durable-execution
  aws iam delete-role --role-name lambda-execution
  ```
  Verified present via `describe-stacks` (StackStatus: CREATE_COMPLETE) and
  `get-role` (CreateDate: 2026-07-30T18:06:25+00:00) immediately before the
  blocked delete attempts. Only `examples/template_external.yaml` was deployed;
  the 46-example harness was never run.
- CI evidence is read-only: GitHub Actions run 30516632123 (`gh run view
  30516632123 --log`). Nothing in 352587389722 was mutated.
