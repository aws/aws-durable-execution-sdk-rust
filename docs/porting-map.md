# JS-to-Rust example porting map

This document tracks how the JavaScript SDK's `examples/` tree maps onto the
Rust `examples/` tree: which JS examples are ported one-to-one, which are
covered by a representative example of the same pattern, and which do not
apply to Rust (with the reason). It exists so future contributors can see at
a glance why an example does or does not have a Rust counterpart.

Ground rule: every example is a single honest workload for one pattern. Where
the JS repo ships several near-identical variants of one pattern, the Rust
tree ports one representative and records the rest as covered by it; that is
deliberate, not a coverage gap.

## Family plan

The JS examples are grouped into three families so coverage gaps stay
visible per family.

| Family | Scope | Status |
| --- | --- | --- |
| **1: basics** | fundamental `step`, `wait`, retry, and no-op handler patterns | **implemented** |
| **2: coordination & fan-out** | `run_in_child_context`, `parallel`, `map`, combinators (`try_join_all`/`join_all`/`select_ok`/`race`), concurrent fan-out via `.spawn()`, determinism/replay behaviors | **implemented** |
| **3: external, serdes & cross-cutting** | `invoke`, `create_callback`, `wait_for_callback`, `wait_for_condition`, serdes, large payloads, `tracing` logging, and the capstone comprehensive example | **implemented** |

## Mapping summary

- **Total JS examples:** 133 (the 135 non-test `.ts` handler files under the JS
  `examples/` tree, minus 2 shared helpers: `shared/uppercase-serdes.ts` and
  `otel/shared/otel-test-setup.ts`, which are infra, not examples).
- **Mapped (ported or port-planned):** 106.
- **Skipped:** 27. Grouped reasons:
  - **10**: `otel/*`: the Rust SDK has no OpenTelemetry plugin. `tracing` is
    the logging story; a plugin system is a
    possible future addition.
  - **4**: `force-checkpointing/*`: there is no force-checkpoint API in the
    Rust SDK's public surface.
  - **2**: `*virtual*` (`run-in-child-context/virtual`,
    `run-in-child-context/serdes-virtual`): the user-facing `virtualContext`
    child-context concept is not part of the Rust SDK's public API. (The
    map/parallel `virtual-context` examples exercise a different concept, FLAT
    nesting mode, which the Rust SDK does have; they are port-planned in the
    coordination family via `NestingMode::Flat`.)
  - **3**: `context-validation/*`: nesting durable ops inside a step or a
    `wait_for_condition` check is a **compile error** in Rust (`StepContext`
    exposes no durable operations), and using a parent context from a child is
    caught by the runtime task-ownership guard: these are negative tests the
    Rust type system / ownership detector prevent by construction, not runnable
    usage examples.
  - **2**: `child-operations-invalid-depth`, `child-operations-preservation`:
    both exercise `pluginsConfig.childOperationsDepth`, a JS plugin config with
    no analogue in the Rust SDK (a plugin system is a possible future addition).
  - **2**: `logger-test/powertools-logger`, `logger-test/simple-powertools-logger`:
    AWS Lambda Powertools is JS-specific; there is no Powertools for Rust, and
    `tracing` is the ecosystem-standard replacement.
  - **1**: `parallel/custom-summary-generator`: no summary-generator hook in
    the Rust SDK's public surface.
  - **1**: `map-completion-config-issue`: a JS-specific bug reproduction, not
    a usage example.
  - **1**: `wait/unawaited`: relies on JS promise **eagerness** (an unawaited
    op still makes progress). Rust builders are **lazy** by design:
    an unawaited builder does nothing. The eager analogue is `.spawn()`,
    demonstrated in the coordination family, so this specific example does not
    port, but the capability it shows is not lost.
  - **1**: `promise/unhandled-rejection`: JS-native unhandled-promise-rejection
    semantics. Rust is `Result`-based with no floating rejections; there is no
    analogous failure mode.

## Family 1: basics

Seven representative workloads. The "covers" rows are JS variants of the same
pattern collapsed into a representative per the one-workload-per-pattern rule.

| JS example | Disposition | Rust target |
| --- | --- | --- |
| `hello-world` | ✅ ported | `basics/hello_world` |
| `simple-execution` | ✅ covered | by `basics/hello_world` (handler with no durable ops) |
| `non-durable` | ✅ covered | by `basics/hello_world` (non-durable handler shape) |
| `step/basic` | ✅ ported | `basics/step_basic` |
| `step/named` | ✅ ported | `basics/step_named` |
| `step/with-retry` | ✅ ported | `basics/step_with_retry` |
| `step/steps-with-retry` | ✅ covered | by `basics/step_with_retry` (JS adds a DDB poll loop: scaffolding omitted per the one-workload rule) |
| `step/attempt-fallback` | ✅ covered | by `basics/step_with_retry` (both key off `StepContext::attempt()`) |
| `wait/basic` | ✅ ported | `basics/wait_basic` |
| `wait/named` | ✅ ported | `basics/wait_named` |
| `wait/configurable` | ✅ covered | by `basics/wait_basic` (duration-via-payload is a trivial variant) |
| `multiple-waits` | ✅ ported | `basics/multiple_waits` |
| `wait/unawaited` | ⊘ skip | Rust builders are lazy by design; the eager analogue is `.spawn()` (coordination family) |

## Family 2: coordination & fan-out

Nineteen representative workloads under `examples/coordination/`. Per the
one-workload-per-pattern rule, near-identical JS variants of one pattern are
collapsed into a representative and marked "covered by" it; every JS row below
is therefore either ported or covered, none left planned. Cross-cutting notes:

- The four combinators accept any [`DurableFuture`], so the JS `*-wait` variants
  (a combinator over waits) are the same code with a wait operand: covered by
  the base combinator example.
- The `.spawn()` fan-out mechanic is identical regardless of the operation
  inside, so the `concurrent/*` variants are covered by one representative;
  callbacks and invoke themselves are demonstrated in Family 3.
- `parallel`/`map` completion is one [`CompletionConfig`] surface (`min_successful`
  plus the tolerated-failure knobs). `map_completion` demonstrates a tolerated
  failure returning the partial `Vec` of successes; `parallel_completion`
  demonstrates `min_successful` early completion. Note: with a
  tolerated failure, `map`'s plain `Vec` await returns the successful subset,
  while `parallel`'s plain `Vec` await surfaces the tolerated branch error (it
  cannot yield a complete `Vec`); the completion policy itself is identical.

| JS example | Disposition | Rust target / note |
| --- | --- | --- |
| `run-in-child-context/basic` | ✅ ported | `coordination/child_basic` |
| `run-in-child-context/with-failing-step` | ✅ ported | `coordination/child_failing_step` |
| `run-in-child-context/error-data-propagation` | ✅ covered | by `child_failing_step` (error + message cross the child boundary intact) |
| `run-in-child-context/checkpoint-size-limit` | ✅ covered | by `child_large_data` for the inline case; the above-inline-size overflow path is `external/large_payload` (FileSystemSerdes overflow, Family 3) |
| `run-in-child-context/large-data` | ✅ ported | `coordination/child_large_data` |
| `run-in-child-context/serdes` | ✅ ported | `coordination/child_serdes` |
| `run-in-child-context/serdes-large-payload` | ✅ covered | by `child_serdes` (same per-operation serdes seam; large payloads use it unchanged) |
| `run-in-child-context/virtual` | ⊘ skip | no virtual context in the Rust SDK's public API |
| `run-in-child-context/serdes-virtual` | ⊘ skip | no virtual context in the Rust SDK's public API |
| `block-example` | ✅ covered | by `child_basic` (sequential child-context composition; no distinct API) |
| `parallel/basic` | ✅ ported | `coordination/parallel_basic` |
| `parallel/empty` | ✅ ported | `coordination/parallel_empty` |
| `parallel/wait` | ✅ covered | by `parallel_basic` (a branch performs a durable wait) |
| `parallel/invoke` | ✅ covered | by `parallel_basic` (branch bodies are ordinary durable code; `invoke` itself is `external/invoke_*`, Family 3) |
| `parallel/heterogeneous` | ✅ ported | `coordination/parallel_heterogeneous`: held builders + `tokio::join!` (lazy-builder composition) |
| `parallel/error-preservation` | ✅ covered | by `combinator_join_all` (each failure preserved as `Settled::Rejected`) |
| `parallel/min-successful` | ✅ covered | by `parallel_completion` (`CompletionConfig::with_min_successful`) |
| `parallel/min-successful-with-callback` | ✅ covered | by `parallel_completion` (min_successful; callback is Family 3) |
| `parallel/min-successful-with-passing-threshold` | ✅ covered | by `parallel_completion` (same completion surface) |
| `parallel/should-complete` | ✅ covered | by `parallel_completion` (`CompletionConfig`) |
| `parallel/failure-threshold-exceeded-count` | ✅ covered | by `parallel_completion` (`CompletionConfig` tolerated-failure thresholds) |
| `parallel/failure-threshold-exceeded-percentage` | ✅ covered | by `parallel_completion` (`tolerated_failure_percentage` field) |
| `parallel/tolerated-failure-count` | ✅ covered | by `parallel_completion` (`CompletionConfig`); tolerated-failure return semantics shown in `map_completion` |
| `parallel/tolerated-failure-percentage` | ✅ covered | by `parallel_completion` (`CompletionConfig`) |
| `parallel/custom-summary-generator` | ⊘ skip | no summary-generator hook in the Rust SDK's public API |
| `parallel/virtual-context` | ✅ ported | `coordination/parallel_virtual`: `.nesting(NestingMode::Flat)` |
| `map/basic` | ✅ ported | `coordination/map_basic` (also demonstrates `.item_namer()`) |
| `map/empty` | ✅ ported | `coordination/map_empty` |
| `map/large-scale` | ✅ covered | by `map_concurrency` (bounded fan-out stands in for scale) |
| `map/high-concurrency-invoke` | ✅ covered | by `map_concurrency` (`.max_concurrency()`) |
| `map/error-preservation` | ✅ covered | by `map_completion` (tolerated failure's error preserved) |
| `map/error-type-preservation-replay` | ✅ covered | by `map_completion` (error preserved across replay) |
| `map/min-successful` | ✅ covered | by `map_completion` (`CompletionConfig`) |
| `map/failure-threshold-exceeded-count` | ✅ covered | by `map_completion` (`CompletionConfig`) |
| `map/failure-threshold-exceeded-percentage` | ✅ covered | by `map_completion` (`tolerated_failure_percentage` field) |
| `map/tolerated-failure-count` | ✅ ported | `coordination/map_completion` (`CompletionConfig::with_tolerated_failure_count`) |
| `map/tolerated-failure-percentage` | ✅ covered | by `map_completion` (same completion surface) |
| `map/virtual-context` | ✅ ported | `coordination/map_virtual`: `.nesting(NestingMode::Flat)` |
| `map-completion-config-issue` | ⊘ skip | JS-specific bug reproduction, not a usage example |
| `promise/all` | ✅ ported | `coordination/combinator_try_join_all` (= `Promise.all`) |
| `promise/all-settled` | ✅ ported | `coordination/combinator_join_all` (= `Promise.allSettled`) |
| `promise/any` | ✅ ported | `coordination/combinator_select_ok` (= `Promise.any`) |
| `promise/race` | ✅ ported | `coordination/combinator_race` (= `Promise.race`) |
| `promise/combinators` | ✅ covered | by the four combinator examples above |
| `promise/all-wait` | ✅ covered | by `combinator_try_join_all` (combinators accept any `DurableFuture`, incl. waits) |
| `promise/race-wait` | ✅ covered | by `combinator_race` (a race over wait operands) |
| `promise/replay` | ✅ covered | by `combinator_race` (checkpointed-winner replay, documented in the example) |
| `promise/unhandled-rejection` | ⊘ skip | JS-native unhandled-rejection semantics; no `Result`-based analogue |
| `concurrent/operations` | ✅ ported | `coordination/concurrent_operations` (`.spawn()` fan-out) |
| `concurrent/wait` | ✅ covered | by `concurrent_operations` (same `.spawn()` mechanic with a wait) |
| `concurrent/callback-wait` | ✅ covered | by `concurrent_operations` (fan-out mechanic; callbacks are Family 3) |
| `concurrent/callback-submitter` | ✅ covered | by `concurrent_operations` (fan-out mechanic; callbacks are Family 3) |
| `context-validation/parent-context-in-step` | ⊘ skip | compile error in Rust (`StepContext` exposes no durable ops) |
| `context-validation/parent-context-in-wait-condition` | ⊘ skip | compile error in Rust (check fn receives `StepContext`) |
| `context-validation/parent-context-in-child` | ⊘ skip | caught by the runtime task-ownership guard, not a success-path example |

[`CompletionConfig`]: https://docs.rs/aws-durable-execution-sdk-rust
[`DurableFuture`]: https://docs.rs/aws-durable-execution-sdk-rust

## Family 3: external, serdes & cross-cutting

Twenty-two representative workloads under `examples/external/` (twenty
user-facing examples plus two chained-invoke companion callees,
`invoke_target` and `invoke_target_tenant`). Per the one-workload-per-pattern
rule, near-identical JS variants of one pattern are collapsed into a
representative and marked "covered by" it; every JS row below is therefore
ported or covered, none left planned. Smoke notes:

- **Callbacks are driven externally.** After the execution suspends on the
  callback, the smoke harness completes it with
  `aws lambda send-durable-execution-callback-success`, and the execution then
  finishes SUCCEEDED.
- **Expected-FAILED examples.** `create_callback_timeout` and
  `wait_for_callback_timeout` catch the timeout and finish SUCCEEDED;
  `handler_error` returns an error from the handler and finishes FAILED
  deterministically: a FAILED terminal state is its PASS condition.
- **Companion callees.** `invoke_target` / `invoke_target_tenant` are deployed
  in the same stack; callers reach them through the `TARGET_FUNCTION_NAME`
  environment variable (`${Target.Arn}:$LATEST`).
- **Orthogonal variants.** Retry, serdes, error-instance, mixed-op, and
  nesting variants are chain-method or composition variations of a base
  operation, so they are covered by the base example rather than re-deployed.

| JS example | Disposition | Rust target / note |
| --- | --- | --- |
| `invoke/simple` | ✅ ported | `external/invoke_simple` (companion callee `external/invoke_target`) |
| `invoke/tenant-id` | ✅ ported | `external/invoke_tenant_id` (`.tenant_id()`) |
| `invoke/tenant-target` | ✅ ported | `external/invoke_target_tenant` (PER_TENANT callee) |
| `with-retry/invoke` | ✅ covered | by `invoke_simple` (retry is an orthogonal chain method; exhaustion shown in `retry_exhaustion`) |
| `with-retry/invoke-target` | ✅ covered | by `invoke_target` (identical echo callee) |
| `with-retry/callback` | ✅ covered | by `create_callback_simple` + `retry_exhaustion` (retry is an orthogonal chain method) |
| `create-callback/simple` | ✅ ported | `external/create_callback_simple` |
| `create-callback/concurrent` | ✅ covered | by `coordination/concurrent_operations` (identical `.spawn()` fan-out mechanic) |
| `create-callback/heartbeat` | ✅ ported | `external/create_callback_heartbeat` (`.heartbeat()`) |
| `create-callback/timeout` | ✅ ported | `external/create_callback_timeout` (caught timeout → SUCCEEDED) |
| `create-callback/failures` | ✅ covered | by `create_callback_timeout` (callback failure/error path) |
| `create-callback/error-instance` | ✅ covered | by `create_callback_timeout` (`CallbackError` carries the failure instance) |
| `create-callback/mixed-ops` | ✅ covered | by `external/comprehensive` (callbacks composed with other operations) |
| `create-callback/serdes` | ✅ covered | `create_callback` exposes `.serdes(...)` (deserialize-only; the payload is delivered externally); it is the same `Serdes` seam `serde_basic` demonstrates |
| `wait-for-callback/basic` | ✅ ported | `external/wait_for_callback_basic` |
| `wait-for-callback/anonymous` | ✅ covered | by `wait_for_callback_basic` (anonymous = the no-op submitter) |
| `wait-for-callback/child-context` | ✅ covered | by `wait_for_callback_basic` + `coordination/child_basic` |
| `wait-for-callback/nested` | ✅ covered | by `wait_for_callback_basic` (nesting is child-context composition) |
| `wait-for-callback/timeout` | ✅ ported | `external/wait_for_callback_timeout` (caught timeout → SUCCEEDED) |
| `wait-for-callback/failures` | ✅ covered | by `wait_for_callback_timeout` (failure/timeout error path) |
| `wait-for-callback/failing-submitter` | ✅ covered | by `wait_for_callback_submitter` (submitter error drives `submitter_retry`) |
| `wait-for-callback/submitter-failure-catchable` | ✅ covered | by `wait_for_callback_submitter` (catchable submitter error) |
| `wait-for-callback/submitter-retry-success` | ✅ ported | `external/wait_for_callback_submitter` (`.submitter_retry()`) |
| `wait-for-callback/heartbeat-sends` | ✅ covered | by `create_callback_heartbeat` (same `.heartbeat()` surface) |
| `wait-for-callback/mixed-ops` | ✅ covered | by `external/comprehensive` |
| `wait-for-callback/multiple-invocations` | ✅ covered | by `wait_for_callback_basic` (suspend/resume across invocations is intrinsic) |
| `wait-for-callback/quick-completion` | ✅ covered | by `wait_for_callback_basic` (completion before suspend is the same path) |
| `wait-for-callback/serdes` | ✅ covered | `wait_for_callback` exposes `.serdes(...)`, threaded through to the inner callback decode; same `Serdes` seam as `serde_basic` |
| `wait-for-callback/error-instance-failure` | ✅ covered | by `wait_for_callback_timeout` |
| `wait-for-callback/error-instance-submitter` | ✅ covered | by `wait_for_callback_submitter` |
| `wait-for-callback/error-instance-timeout` | ✅ covered | by `wait_for_callback_timeout` |
| `wait-for-condition` | ✅ ported | `external/wait_for_condition` |
| `serde/basic` | ✅ ported | `external/serde_basic` (custom `Serdes`) |
| `serde/configure-serdes/configure-serdes` | ✅ ported | `external/serde_configure` (`Options` serdes) |
| `serde/configure-serdes/configure-callback-deserializer` | ✅ covered | no separate callback-deserializer channel in this SDK; the callback decode falls back to the execution-wide `Options::serdes` (overridable per-op via `.serdes(...)`), which `serde_configure` exercises |
| `large-payload` | ✅ ported | `external/large_payload` (FileSystemSerdes overflow) |
| `logger-test/log-levels` | ✅ ported | `external/logging_levels` (`tracing`) |
| `logger-test/after-wait` | ✅ ported | `external/logging_after_wait` (replay-suppressed `tracing`) |
| `logger-test/after-callback` | ✅ covered | by `logging_after_wait` (replay-suppressed emission around a suspend point) |
| `logger-test/powertools-logger` | ⊘ skip | no Powertools for Rust; `tracing` is the replacement |
| `logger-test/simple-powertools-logger` | ⊘ skip | no Powertools for Rust; `tracing` is the replacement |
| `otel/basic-steps` | ⊘ skip | no OpenTelemetry plugin in the Rust SDK (plugin system is a follow-on; `tracing` is the logging story) |
| `otel/callback` | ⊘ skip | no OpenTelemetry plugin in the Rust SDK |
| `otel/child-context` | ⊘ skip | no OpenTelemetry plugin in the Rust SDK |
| `otel/combined` | ⊘ skip | no OpenTelemetry plugin in the Rust SDK |
| `otel/invoke` | ⊘ skip | no OpenTelemetry plugin in the Rust SDK |
| `otel/log-enrichment` | ⊘ skip | no OpenTelemetry plugin in the Rust SDK |
| `otel/retry-steps` | ⊘ skip | no OpenTelemetry plugin in the Rust SDK |
| `otel/wait-and-resume` | ⊘ skip | no OpenTelemetry plugin in the Rust SDK |
| `otel/wait-for-condition` | ⊘ skip | no OpenTelemetry plugin in the Rust SDK |
| `otel/xray-e2e` | ⊘ skip | no OpenTelemetry plugin in the Rust SDK |
| `comprehensive-operations` | ✅ ported | `external/comprehensive` (capstone: all operations) |
| `error-determinism` | ✅ ported | `external/error_determinism` (determinism contract) |
| `step/step-error-determinism` | ✅ covered | by `error_determinism` (same determinism contract applied to step errors) |
| `no-replay-execution` | ✅ ported | `external/no_replay_execution` (`is_replaying`) |
| `undefined-results` | ✅ ported | `external/optional_results` (Rust `Option`/`()` for JS `undefined`) |
| `handler-error` | ✅ ported | `external/handler_error` (handler-level error; terminal FAILED is the PASS condition) |
| `retry-exhaustion` | ✅ ported | `external/retry_exhaustion` (retry until `RetryDecision::Stop`) |
| `step/interrupted-no-retry` | ✅ covered | by the step conformance suite (`AtMostOncePerRetry` via `.semantics()`); a runnable example needs a forced mid-step Lambda timeout to observe the started-but-not-retried path, which a bounded smoke test cannot reproduce |
| `force-checkpointing/step-retry` | ⊘ skip | no force-checkpoint API in the Rust SDK's public API |
| `force-checkpointing/callback` | ⊘ skip | no force-checkpoint API in the Rust SDK's public API |
| `force-checkpointing/invoke` | ⊘ skip | no force-checkpoint API in the Rust SDK's public API |
| `force-checkpointing/multiple-wait` | ⊘ skip | no force-checkpoint API in the Rust SDK's public API |
| `child-operations-invalid-depth` | ⊘ skip | `pluginsConfig.childOperationsDepth`: plugin config not in the Rust SDK |
| `child-operations-preservation` | ⊘ skip | `pluginsConfig.childOperationsDepth`: plugin config not in the Rust SDK |
