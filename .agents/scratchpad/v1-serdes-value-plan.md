# V1 — Serdes takes the typed value erased to `serde_json::Value`

Planning slice. **No behavioural change.** The only file added is this one.
Every line/file reference below was read at `11376ed` (branch `alpha`).

## Verdict up front

The design is **implementable and object-safe**, and it does collapse the two
input-shape rules into one. Object safety is *compiled*, not assumed (§3).

But it is **not** a drop-in, and two of the problems are not cosmetic:

1. **`serde_json::to_value` is not a faithful stand-in for
   `serde_json::to_string`.** Struct field order is lost (`Value::Object` is a
   `BTreeMap` without the `preserve_order` feature), so every struct payload's
   checkpoint bytes change. Measured, §8.1.
2. **`i128`/`u128` values outside `i64`/`u64` range cannot be represented in
   `Value` at all.** `to_string` succeeds, `to_value` returns
   `Err("number out of range")`. A payload type that works today fails after.
   Measured, §8.2.
3. The "skip `Value` when no custom serdes is configured" shortcut (§7)
   reintroduces two internal conversion paths — one public Serdes contract but
   two internal serialization strategies. Whether that is acceptable depends on
   whether "one rule" means "one public API shape for implementors" or "one
   byte-identical representation everywhere." See §7 for the full analysis.

None of these sink the design. (1) is addressable with `preserve_order` for
struct field order only (but does NOT resolve §8.2-8.4). (2) is a narrow
documented limitation. (3) is a decision the owner must make explicitly. They
are all called out plainly rather than designed around, per the slice's
instruction.

## 0. Peer-SDK evidence (read firsthand, not taken on trust)

| SDK | Signature | Location | Receives |
|---|---|---|---|
| Go | `Marshal(ctx context.Context, meta SerdesContext, v any) ([]byte, error)` | `~/github/aws-durable-execution-sdk-go/durable/context.go:96-99` | the value |
| JS | `serialize: (value: T \| undefined, context: SerdesContext) => Promise<string \| undefined>` | `.../src/utils/serdes/serdes.ts:23` (interface at `:23`, `serialize` at `:30-33`) | the value |
| Java | `String serialize(Object value)` | `.../durable/serde/SerDes.java:19` | the value |

All three confirmed at the cited lines. All three receive the **value** and
return a string/bytes. Rust alone receives a pre-rendered JSON string, which is
why its conformance handlers must `serde_json::from_str` before wrapping while
Go's and Python's do not.

Two nuances worth recording, because they bound how far the analogy carries:

- **Go and JS deserialize straight into the target type** (`v any` out-param /
  generic `T`). Java uses a `TypeToken`. None of them has a `Value`
  intermediate. Rust's `-> Value` is *not* the peer shape; it is the
  object-safe stand-in for it. The bridge back to `O` is
  `serde_json::from_value`, which is the extra step no peer SDK pays.
- **Java's asymmetry matches the proposal**: value in, `String` out. So
  `fn serialize(&self, value: &Value, ..) -> Result<String, _>` is the right
  asymmetry, not a wart.

## 1. Call-site inventory

Every point where the SDK invokes a serdes or renders a payload for one.
`serde_json` version is **1.0.151** (`Cargo.lock`), a direct dependency
(`Cargo.toml:61`). `Value` enum: `serde_json-1.0.151/src/value/mod.rs:116`.

### 1a. The trait as it stands (`src/serdes.rs`)

| Line | Method | Shape |
|---|---|---|
| 111 | `pub trait Serdes: Debug + Send + Sync` | — |
| 120 | `serialize_to_string(&self, json_str: &str) -> Result<String, _>` | default: identity |
| 135 | `deserialize_from_string(&self, payload: &str) -> Result<String, _>` | default: identity |
| 155 | `serialize_to_string_with_context(&self, json_str: &str, &SerdesContext)` | default: delegates to 120 |
| 173 | `deserialize_from_string_with_context(&self, payload: &str, &SerdesContext)` | default: delegates to 135 |

The engine **always** calls the `_with_context` pair. The context-free pair
exists only as the delegation target and as the ergonomic override point.

### 1b. Target trait shape: TWO context-required methods

The approved design is a **two-method trait**, not four. The context-free
delegation structure was an ergonomic convenience when the SDK was string-in,
string-out. Under the Value boundary, context is always available at every call
site (the engine builds a `SerdesContext` from `wire_id` + `execution_arn`
before calling), so the delegation indirection adds no value. The trait becomes:

```rust
pub trait Serdes: Debug + Send + Sync {
    /// Serialize a structured value to a wire string.
    ///
    /// `value` is the operation result/input erased to `serde_json::Value`.
    /// The default renders compact JSON (`value.to_string()`).
    fn serialize(&self, value: &serde_json::Value, ctx: &SerdesContext)
        -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        Ok(value.to_string())
    }

    /// Deserialize a wire string back to a structured value.
    ///
    /// `data` is the string previously returned by `serialize`.
    /// The default parses it as JSON.
    fn deserialize(&self, data: &str, ctx: &SerdesContext)
        -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
        Ok(serde_json::from_str(data)?)
    }
}
```

**Why not four methods.** The context-free overrides existed so simple serdes
(like `UppercaseSerdes`) could ignore context. Under the new trait, a simple
serdes can still ignore the `_ctx` parameter — it is `&SerdesContext` with a
lifetime-free borrow, so `fn serialize(&self, v: &Value, _: &SerdesContext)`
is as easy to write. Keeping two extra methods doubles the public surface for
no gain and creates ambiguity about which method FileSystemSerdes should
override (the source of the previous plan's defect).

### 1c. Per-module helpers (the real seam)

No operation module calls the trait directly except through one of these. This
is the key structural fact: **the edit is concentrated in 12 helper functions,
not in the ~30 call sites.**

| Path | Helper | Definition | Calls trait at |
|---|---|---|---|
| Step — serialize | `serialize_value` | `src/step.rs:478` | `step.rs:492` |
| Step — deserialize | `deserialize_result` | `src/step.rs:505` | `step.rs:511` |
| Step — replay | `replay_success` | `src/step.rs:532` | via `deserialize_result`, `step.rs:538` |
| Invoke — input (live, inlined) | — | `src/invoke.rs:103-118` | `invoke.rs:107` |
| Invoke — input (helper) | `serialize_invoke_input` | `src/invoke.rs:142` | `invoke.rs:156` |
| Invoke — result | `deserialize_invoke_result` | `src/invoke.rs:170` | `invoke.rs:176` |
| Callback — decode only | `deserialize_callback_result` | `src/callback.rs:516` | `callback.rs:522` |
| Child — serialize | `serialize_value` | `src/child.rs:272` | `child.rs:282` |
| Child — deserialize | `deserialize_value` | `src/child.rs:290` | `child.rs:297` |
| WFC — serialize state | `serialize_state` | `src/wait_for_condition.rs:353` | `wait_for_condition.rs:365` |
| WFC — deserialize state | `deserialize_state_str` | `src/wait_for_condition.rs:388` | `wait_for_condition.rs:394` |
| Map/parallel — batch item in summary | `serialize_value` | `src/map_parallel.rs:1524` | `map_parallel.rs:1532` |
| Map/parallel — batch item from summary | `deserialize_value` | `src/map_parallel.rs:1545` | `map_parallel.rs:1551` |
| Map/parallel — **ITEM** serialize | `serialize_item_value` | `src/map_parallel.rs:1571` | `map_parallel.rs:1588` |
| Map/parallel — **ITEM** deserialize | `deserialize_item_value` | `src/map_parallel.rs:1603` | `map_parallel.rs:1613` |
| Map/parallel — whole-batch result serialize | inlined `result_serdes` | `src/map_parallel.rs:1052-1057` (and empty-collection twin `:700-705`) | `:1054`, `:702` |
| Map/parallel — whole-batch result deserialize | inlined in `replay_terminal_batch` | `src/map_parallel.rs:1295-1302` | `:1297` |

### 1d. The invoke-input path — COMPLETE inventory

The invoke input is the ONE non-mechanical site. The full path:

1. **`context.rs:638`** — initial serialization at call site:
   ```rust
   let serialized_input = serde_json::to_string(&input).map_err(|e| e.to_string());
   ```
   This produces a `Result<String, String>` and passes it to `InvokeBuilder::new`.

2. **`builders.rs:465`** — `InvokeBuilder` field definition:
   ```rust
   serialized_input: Result<String, String>,
   ```

3. **`builders.rs:487`** — `InvokeBuilder::new` accepts it as a parameter
   (line 487 in the constructor arg list, stored into the field at ~:494).

4. **`invoke.rs:31`** — `InvokeExecution` field:
   ```rust
   pub(crate) serialized_input: Result<String, String>,
   ```

5. **`invoke.rs:47-56`** — early error surfacing: the `Err` arm returns
   `SerializationFailed` before any checkpoint or Lambda call happens.

6. **`invoke.rs:103-118`** — live path: the `Ok(serialized_input)` string is
   handed to the serdes at `:107`.

**Why this path is special:** the user's `I: Serialize` type is erased by
`context.rs:638` *before* the future body. The builder carries the result
across an `.await` boundary. To hand the serdes a `&Value` instead of `&str`,
the builder must carry `Result<serde_json::Value, String>` instead of
`Result<String, String>`.

**The double-serialization problem.** If the builder carries `Value`, the
no-serdes default path must still produce wire text. Two options:

- **(a) Carry only `Value`.** Change `context.rs:638` to
  `serde_json::to_value(&input).map_err(|e| e.to_string())`.
  The default arm at `invoke.rs:116-117` becomes
  `serde_json::to_string(&value)?` (serializes the Value, not the original `I`).
  This produces byte-identical output to `to_string(&input)` **only if
  `preserve_order` is enabled** (otherwise struct field order differs). One
  serialization of the user's type (`to_value`), one rendering of the Value.

- **(b) Carry both `Value` AND `String`.** Change `context.rs:638` to:
  ```rust
  let value = serde_json::to_value(&input);
  let text = serde_json::to_string(&input);
  ```
  The serdes arm uses `value`, the default arm uses `text`. This serializes
  user input **twice**, which (i) doubles the cost at the call site, (ii) can
  produce inconsistent results for types with interior mutability or
  non-deterministic `Serialize` impls (e.g., `HashMap` iteration order). The
  inconsistency is the critical defect: the `Value` handed to the serdes and
  the `String` on the wire can represent different data.

**Recommendation: (a), carry only Value.** The inconsistency risk in (b) is a
correctness hazard. Under (a), `to_string(&value)` on the Value is guaranteed
consistent with what the serdes receives. The struct-field-order change (§8.1)
is the same issue as every other site and is resolved uniformly by
`preserve_order` or accepted as a wire-format migration.

The complete edit for invoke-input:
- `context.rs:638`: `serde_json::to_value(&input).map_err(|e| e.to_string())`
- `builders.rs:465`: `serialized_input: Result<serde_json::Value, String>`
- `builders.rs:487`: parameter type changes to match
- `invoke.rs:31`: `pub(crate) serialized_input: Result<serde_json::Value, String>`
- `invoke.rs:47-56`: `Err` arm unchanged (same pattern match)
- `invoke.rs:103-118`:
  ```rust
  let wire_payload = if let Some(ps) = effective_payload_serdes {
      ps.serialize(&serialized_input, &serdes_ctx)?
  } else {
      serde_json::to_string(&serialized_input)?  // Value -> text
  };
  ```

### 1e. Live and replay call sites, by path

| Path | Live | Replay |
|---|---|---|
| Step | `step.rs:285` (serialize), `:294` (round-trip deserialize) | `step.rs:164` → `replay_success` → `:538` |
| Invoke input | `context.rs:638` (serialize to Value), `invoke.rs:107` (Value through serdes or to wire) | n/a (input written once, before suspend) |
| Invoke result | n/a (backend writes it) | `invoke.rs:69` |
| Callback | n/a — **deserialize only** | `callback.rs:84` |
| Child context | `child.rs:74`, `child.rs:133` | `child.rs` `replay_success:248` → `deserialize_value:297` |
| WFC state | `wait_for_condition.rs:221` (serialize), `:228` (round-trip) | `:140` → `replay_terminal_success:335`; `:186` → `deserialize_state` |
| Map/par ITEM (Flat nesting) | `map_parallel.rs:1131` / `:1132` | `map_parallel.rs:1349` (`replay_terminal_child`) |
| Map/par ITEM (Child nesting) | `map_parallel.rs:1187` / `:1215` | `map_parallel.rs:1349` |
| Map/par item inside batch summary | `map_parallel.rs:1676` (`from_batch_result`) | `map_parallel.rs:1309` (`to_batch_result`) → `deserialize_value` |
| Map/par whole-batch result | `map_parallel.rs:1049-1057`; empty-collection twin `:697-705` | `map_parallel.rs:1295-1302` |

### 1f. How `11376ed`'s item split relates to the target shape

`11376ed` ("fix(serdes): pass raw value to custom item serdes") split the item
paths onto `serialize_item_value` / `deserialize_item_value`
(`map_parallel.rs:1571` / `:1603`). Read closely: it is **this same design,
implemented once, locally, through a `Value` it then throws away.**

`serialize_item_value:1580-1587` already does exactly the erasure the proposal
wants:

```rust
let json_value = serde_json::to_value(value)?;   // erase to Value
let raw = match json_value {
    serde_json::Value::String(inner) => inner,   // ...then flatten to text
    other => other.to_string(),
};
s.serialize_to_string_with_context(&raw, serdes_ctx)
```

Under `fn serialize(&self, &Value, &SerdesContext)`:

- `serialize_item_value` becomes byte-identical to `serialize_value` — the
  `Value::String(inner) => inner` flattening exists *only* to fake "hand over
  the value" across a `&str` boundary. Delete it.
- `deserialize_item_value`'s try/fallback heuristic **disappears entirely**.
  The serdes returns a `Value`; `serde_json::from_value::<O>` is exact.
- The two-rules documentation block at `serdes.rs:29-71` is deleted.

**So `11376ed` is a partial, local application of V1, and V1 subsumes it.**

## 2. Per-site: today → after → the edit

The uniform edit, applied to each of the 12 helpers:

**Serialize side** — today:
```rust
let json_str = serde_json::to_string(value)?;          // O -> text
if let Some(s) = serdes { s.serialize_to_string_with_context(&json_str, ctx) }
else { Ok(json_str) }
```
after:
```rust
if let Some(s) = serdes {
    let v = serde_json::to_value(value)?;              // O -> Value
    s.serialize(&v, ctx)
} else {
    serde_json::to_string(value).map_err(Into::into)   // unchanged default path
}
```

**Deserialize side** — today:
```rust
let json_str = if let Some(s) = serdes {
    s.deserialize_from_string_with_context(payload, ctx)?
} else { payload.to_owned() };
serde_json::from_str(&json_str)
```
after:
```rust
if let Some(s) = serdes {
    let v = s.deserialize(payload, ctx)?;
    serde_json::from_value(v)                           // exact, no guessing
} else {
    serde_json::from_str(payload)                       // unchanged default path
}
```

Site-by-site:

| Site | Receives today | Must receive after | Edit |
|---|---|---|---|
| `step.rs:478` `serialize_value` | JSON text of `O` | `&Value` of `O` | helper body |
| `step.rs:505` `deserialize_result` | wire text → returns text | wire text → returns `Value` | helper body |
| `invoke.rs:103-118` live payload | `String` (from `context.rs:638`) | `Value` (from `context.rs:638`) | §1d edit |
| `invoke.rs:142` `serialize_invoke_input` | JSON text of `I` | `&Value` of `I` | helper body |
| `invoke.rs:170` `deserialize_invoke_result` | wire text | returns `Value` | helper body |
| `callback.rs:516` `deserialize_callback_result` | wire text | returns `Value` | helper body |
| `child.rs:272` / `:290` | JSON text | `&Value` / returns `Value` | helper bodies |
| `wait_for_condition.rs:353` / `:388` | JSON text | `&Value` / returns `Value` | helper bodies |
| `map_parallel.rs:1524` / `:1545` (summary item) | JSON text | `&Value` / returns `Value` | helper bodies |
| `map_parallel.rs:1571` `serialize_item_value` | raw flattened text | **delete; fold into `serialize_value`** | deletion |
| `map_parallel.rs:1603` `deserialize_item_value` | raw text, guessed back | **delete; fold into `deserialize_value`** | deletion |
| `map_parallel.rs:1052-1057` + `:700-705` (batch result) | JSON text | `&Value` | inline edit ×2 |
| `map_parallel.rs:1295-1302` (batch replay) | wire text | returns `Value` | inline edit |

## 3. Object safety — compiled, not assumed

`Box<dyn Serdes>` / `Arc<dyn Serdes>` is stored in **27 places** (mechanically
counted via `grep -rn "Box<dyn Serdes>\|Arc<dyn Serdes>" src/` excluding
comments, doctests, and test modules):

**Options (2):**
- `options.rs:59` — `Options.serdes: Option<Box<dyn Serdes>>`
- `options.rs:123` — `OptionsBuilder.serdes: Option<Box<dyn Serdes>>`

**Context/Arc (3):**
- `context.rs:58` — `default_serdes: Option<Arc<dyn Serdes>>`
- `context.rs:170` — constructor param `Option<Arc<dyn Serdes>>`
- `lib.rs:934` — `let default_serdes: Option<StdArc<dyn Serdes>> = serdes.map(StdArc::from);`

**Execution structs (11):**
- `step.rs:141` — `StepExecution.serdes`
- `invoke.rs:32` — `InvokeExecution.payload_serdes`
- `invoke.rs:33` — `InvokeExecution.result_serdes`
- `child.rs:43` — `ChildExecution.serdes`
- `callback.rs:57` — `CallbackExecution` (variant 1) `.serdes`
- `callback.rs:187` — `CallbackExecution` (variant 2) `.serdes`
- `callback.rs:322` — `CallbackResolveExecution.serdes`
- `wait_for_condition.rs:112` — `WfcExecution.serdes`
- `map_parallel.rs:314` — `MapExecution.serdes`
- `map_parallel.rs:315` — `MapExecution.result_serdes`
- `map_parallel.rs:450` — `ParallelExecution.serdes`
- `map_parallel.rs:451` — `ParallelExecution.result_serdes`

**Function params and runtime wrappers (3):**
- `map_parallel.rs:607` — `execute_batch` param `serdes`
- `map_parallel.rs:608` — `execute_batch` param `result_serdes`
- `map_parallel.rs:754` — `Arc<Option<Box<dyn Serdes>>>` (shared across tasks)

**Builder structs (11):**
- `builders.rs:123` — `StepBuilder.serdes`
- `builders.rs:466` — `InvokeBuilder.payload_serdes`
- `builders.rs:467` — `InvokeBuilder.result_serdes`
- `builders.rs:642` — `ChildBuilder.serdes`
- `builders.rs:827` — `WaitForConditionBuilder.serdes`
- `builders.rs:1067` — `CreateCallbackBuilder.serdes`
- `builders.rs:1197` — `WaitForCallbackBuilder.serdes`
- `builders.rs:1392` — `MapBuilder.serdes`
- `builders.rs:1393` — `MapBuilder.result_serdes`
- `builders.rs:1672` — `ParallelBuilder.serdes`
- `builders.rs:1673` — `ParallelBuilder.result_serdes`

Total: 2 + 3 + 11 + 3 + 11 = **30** (including runtime wrappers and params;
**27** if counting only persistent struct fields + the Arc conversion).

**Compiled proof.** Scratch crate outside the repo at `/tmp/serdes-objsafe-v1`
(edition 2024, `serde_json 1`), `cargo check` passes. Confirms:

1. `Box<dyn Serdes>` with the two-method proposed signature — **object-safe**.
2. `Option<Box<dyn Serdes>>` as a struct field + `.as_deref()` to
   `Option<&dyn Serdes>` — the Options / per-op-config shape.
3. `Arc<dyn Serdes>` — the `context.rs:58` default_serdes shape.
4. A `FileSystemSerdes` analogue behind the trait object: receives `&Value`,
   writes a file, returns a pointer string. Round-trip asserted.
5. Default method bodies that reference `Value` do not break object safety
   (variant: `Passthrough` implementing zero methods — compiles fine because
   default bodies are not generic).

Why the rejected alternatives stay rejected:
- `fn serialize<T: Serialize>(&self, v: &T)` — generic method → not
  dyn-compatible (`E0038`). All 30 storage sites break.
- `&dyn Any` + downcast — removed by prior finding; must not return.
- `erased_serde` — `scripts/check-direct-deps.sh:17` pins exactly 8 direct
  crates; adding any new one fails `make check`.

## 4. `FileSystemSerdes` (`src/serdes.rs:420-690`)

Under the two-method trait, `FileSystemSerdes` implements BOTH methods WITH
context (since the trait has only `serialize` and `deserialize`, both with
`&SerdesContext`). No context-free stubs needed or possible.

### Today's stubs and their fate

| Method | Today | After |
|---|---|---|
| `serialize_to_string` `:637-645` | stub: returns input unchanged, **never writes** | **Deleted.** Trait no longer has this method. |
| `deserialize_from_string` `:647-665` | envelope sniff + dummy-context hack | **Deleted.** Trait no longer has this method. |
| `serialize_to_string_with_context` `:667` | delegates to inherent `serialize_with_context(json_str, ctx)` | Becomes `fn serialize(&self, value: &Value, ctx: &SerdesContext)` — delegates to the inherent method. |
| `deserialize_from_string_with_context` `:675` | delegates to inherent `deserialize_with_context` | Becomes `fn deserialize(&self, data: &str, ctx: &SerdesContext)` — delegates to the inherent method. |

### Inherent methods — what changes

- **`serialize_with_context` (`:476-518`)**: signature becomes
  `(&self, value: &serde_json::Value, ctx: &SerdesContext) -> Result<String, _>`.

  - `Always` mode: `write_to_file(&serde_json::to_string(value)?, ctx)` →
    return `{"file":"<path>"}`. **Unchanged on the wire.**

  - `Overflow` mode gets materially simpler: the current body (`:487-502`)
    probes `serde_json::from_str::<Value>(json_str).is_ok()` to decide between
    `{"data":...}` and `{"raw":"..."}` envelopes. With a `Value` in hand that
    ambiguity cannot arise — **the `{"raw":...}` write-path variant is deleted**,
    and `{"data": <value>}` is always correct.

- **`deserialize_with_context` (`:520-558`)**: returns `Value` instead of
  `String`.
  - `"file"` arm: `serde_json::from_str(&std::fs::read_to_string(path)?)?`
  - `"data"` arm: `parsed.get("data").cloned().unwrap_or(Value::Null)`
    (replacing `data.to_string()` at `:552`, which re-rendered a Value back to
    text only for the caller to re-parse it).
  - `"raw"` arm (`:544-547`): **RETAINED on read, even though the writer stops
    producing it.** Executions checkpointed before V1 may contain `{"raw":"X"}`
    envelopes. Maps to `Value::String("X")`. Deleting the reader with the
    writer is the obvious mistake.

- **`write_to_file` (`:560-590`)**: unchanged — it already takes `&str`.

**Net:** no error-returning stubs, no context-free path ambiguity, one fewer
envelope variant written, one fewer text↔Value bounce per read. No path where
FileSystemSerdes silently fails to persist. §3's scratch variant 4 compiles
this shape.

## 5. Precedence chain

Wanted: per-operation `.serdes(...)` → execution-wide `Options::serdes` →
plain `serde_json`.

Expressed today as one idiom, 14 sites:
`self.serdes.as_deref().or_else(|| self.ctx.default_serdes())` —
`step.rs:165,252`; `child.rs:76,81,95,135,173`; `callback.rs:85`;
`invoke.rs:72,105`; `wait_for_condition.rs:141,187,222,229`.

**Expressible unchanged: yes, trivially.** The chain resolves to an
`Option<&dyn Serdes>` *before* any helper is called and is entirely independent
of the trait's argument types. V1 does not touch a single one of those 14 lines.
Third link ("plain `serde_json`") = the `None` arm inside each helper.

**Map/parallel exclusion:** `grep -c default_serdes src/map_parallel.rs` = 0.
Map and parallel never consult the execution-wide default. Their chain is
two links: per-operation `.serdes(...)` / `.result_serdes(...)` → plain
`serde_json`. This is deliberate and documented (`options.rs:135-136`).
V1 does not change this. Worth a follow-up slice to close the gap once the
input-shape justification for it is removed.

## 6. Public surface delta

### 6a. Breaking (SemVer-major; pre-alpha per `README.md`, so acceptable)

`src/serdes.rs` — four trait methods become two:

| Before | After |
|---|---|
| `serialize_to_string(&self, &str) -> Result<String>` | **Deleted** |
| `deserialize_from_string(&self, &str) -> Result<String>` | **Deleted** |
| `serialize_to_string_with_context(&self, &str, &SerdesContext) -> Result<String>` | `serialize(&self, &serde_json::Value, &SerdesContext) -> Result<String>` |
| `deserialize_from_string_with_context(&self, &str, &SerdesContext) -> Result<String>` | `deserialize(&self, &str, &SerdesContext) -> Result<serde_json::Value>` |

Also public: `FileSystemSerdes::serialize_with_context` (`:476`) and
`::deserialize_with_context` (`:520`) change signature (§4).

**`serde_json::Value` becomes part of the public API surface.** Consider
re-exporting (`pub use serde_json::Value;`).

**Unchanged:** `SerdesContext`, `FileSystemSerdesMode`,
`FileSystemPathEncoding`, `FileSystemSerdesConfig(Builder)`,
`FileSystemSerdesError`, every `.serdes(...)` / `.payload_serdes(...)` /
`.result_serdes(...)` builder method, `Options::builder().serdes(...)`.
Signature is `impl Serdes + 'static` throughout — those compile untouched.

### 6b. `compliance/` — 10 custom serdes, all must be updated

Each needs its method bodies retyped (Value in / Value out). Mechanically
counted via `grep -rn "impl.*Serdes for" compliance/ --include="*.rs"` excluding
`target/` directories. Some use `impl durable::Serdes for` (qualified path),
some use `impl Serdes for` (with a `use` import).

| # | Path | Methods impl'd today |
|---|---|---|
| 1 | `compliance/step/step_custom_serdes/src/main.rs:10` | serialize + deserialize |
| 2 | `compliance/invoke/invoke_custom_payload_serdes/src/main.rs:12` | serialize + deserialize |
| 3 | `compliance/invoke/invoke_custom_result_serdes/src/main.rs:17` | serialize (identity) + deserialize |
| 4 | `compliance/child/child_custom_serdes/src/main.rs:9` | serialize + deserialize |
| 5 | `compliance/wait_for_condition/wait_for_condition_custom_serdes/src/main.rs:12` | serialize + deserialize |
| 6 | `compliance/callback/callback_serdes_happy/src/main.rs:39` | **deserialize ONLY** |
| 7 | `compliance/map/map_custom_serde/src/main.rs:16` | serialize + deserialize |
| 8 | `compliance/map/map_op_serde/src/main.rs:14` | serialize + deserialize |
| 9 | `compliance/map/map_op_serde_replay/src/main.rs:20` | serialize + deserialize |
| 10 | `compliance/parallel/parallel_custom_serde/src/main.rs:10` | serialize + deserialize |

**Note on `callback_serdes_happy` (#6):** it implements `deserialize_from_string`
only — no `serialize_to_string` override. Under the new trait it will implement
only `fn deserialize(&self, data: &str, ctx: &SerdesContext)`. The default
`serialize` (which renders `value.to_string()`) applies, which is correct since
callbacks are deserialize-only paths.

### 6c. `examples/` — 3 custom serdes

| # | Path |
|---|---|
| 8 | `examples/coordination/child_serdes/src/main.rs:24` |
| 9 | `examples/external/serde_basic/src/main.rs:19` |
| 10 | `examples/external/serde_configure/src/main.rs:19` |

### 6d. In-tree doctests and test serdes (also break)

- Doctests in `src/serdes.rs`: `:38-68` (`WrapSerdes`), `:81-107`
  (`UppercaseSerdes`).
- Doctests: `options.rs:148-149`, `builders.rs:534-535`.
- `src/serdes.rs` `test_support` (`:790+`): `HexEnvelopeSerdes`,
  `RecordingSerdes`. `RecordingSerdes` records the exact `&str` / `&Value`
  handed to serialize — its assertions change.
- Inline test serdes: `step.rs:645,648,779,782`; `invoke.rs:445,448,624,627,
  683,686`; `callback.rs:1295`.
- `serdes.rs:1180` `deserialize_from_string_handles_envelope` — **deleted**
  (method no longer exists).
- Prose to delete: `serdes.rs:29-71` (two-rules block),
  `map_parallel.rs:1553-1569` + `:1591-1601` (item-vs-value docs).

## 7. The allocation question, honestly

`serde_json::to_value` costs strictly more than `to_string`: it allocates the
full `Value` tree, *then* the wire `String` is produced from it. At 256 KB
that is roughly 2-3× peak transient memory plus allocator traffic.

**Can the default path avoid it?** Yes — in the `serdes: None` arm of each
helper, we can keep calling `serde_json::to_string(value)` / `from_str::<O>`
directly and never build a `Value`. That is what §2's `else` arms do.

**Does this reintroduce two rules?** This is the crux.

The correct framing (addressing the reviewer's point about §8.2-8.4): there
are actually **two separate concerns** here:

1. **The public Serdes contract** — a single shape. Implementors always
   receive `&Value` and return `Value`. No two-rules documentation needed.
   The §1f defect (raw vs. quoted input) is genuinely eliminated regardless
   of the shortcut.

2. **The internal wire-format representation** — potentially two code paths.
   `to_string(&O)` and `to_string(&to_value(&O))` are NOT byte-identical in
   all cases, even with `preserve_order`:
   - `preserve_order` fixes **struct field order only** (§8.1).
   - **128-bit integers** still diverge: `to_string` succeeds, `to_value`
     errors (§8.2).
   - **Duplicate keys** collapse under `to_value` (§8.3).
   - **`RawValue`** loses its exact text (§8.4, if feature-gated on by a
     downstream crate).

   So `preserve_order` does NOT make the shortcut "a pure optimization with no
   observable consequence." It narrows but does not eliminate the divergence.

**Three options, reframed around the correct distinction:**

- **(A) One public contract, two internal paths.** The `None` arm uses
  `to_string` directly; the `Some(s)` arm builds a `Value`. The public API is
  uniform. The wire bytes may differ between "no serdes" and "identity serdes"
  configurations for the edge cases in §8.2-8.4. Cost: lowest. Risk:
  attaching/detaching a serdes on an in-flight execution changes checkpoint
  bytes for the affected edge cases.

- **(B) One public contract, one internal path.** Every path goes through
  `Value`, even `None`. Wire bytes are uniform. Cost: the §7 allocation on
  100% of operations. The §8.2 limitation applies universally (128-bit payload
  fields that work today start failing). No shortcut complexity.

- **(C) (A) + `preserve_order`.** Same as (A) but struct field order no longer
  diverges. Narrows the divergence set to §8.2-8.4 only — genuinely exotic
  edge cases (128-bit integers, duplicate keys, downstream-enabled `RawValue`).
  Cost: `indexmap`, `equivalent`, `hashbrown` enter as transitive deps of
  `serde_json`. Both dependency gates (`check-direct-deps.sh` and
  `deny.toml [licenses]`) pass — verified. `Value::Object` changes to
  `IndexMap` globally (semver-visible for direct `Value::Object` consumers).

**Recommendation: (A) is sufficient, (C) is nice-to-have.** The divergence
cases under (A) are so narrow (128-bit int fields, duplicate keys, downstream
`RawValue`) that they are documentable limitations rather than correctness
defects. The public contract is single and clear. The shortcut is invisible to
serdes authors. The key invariant — "live and replay agree" — holds within each
configuration because the same code path is used for write and read. The only
scenario where it breaks is changing serdes configuration mid-execution, which
is an explicit misconfiguration.

## 8. What does not fit

Measured in scratch crates outside the repo, comparing `to_string(&v)` against
`to_string(&to_value(&v))` for the same value.

### 8.1 Struct field order is lost without `preserve_order`

```
struct FieldOrder { zebra: u8, apple: u8, mango: u8 }
to_string      = {"zebra":1,"apple":2,"mango":3}     // declaration order
to_value->str  = {"apple":2,"mango":3,"zebra":1}     // sorted (BTreeMap)
```

With `preserve_order`: byte-identical.

### 8.2 `i128`/`u128` beyond 64-bit range — CANNOT be represented in Value

```
struct Big { v: i128 }
Big { v: i128::MAX }   to_string = {"v":170141183460469231731687303715884105727}
                       to_value  = ERR(number out of range)
```

**This is the answer to "can Value represent everything the current text
boundary can?" — no.** `preserve_order` does NOT fix this. It is a real
capability regression for types with out-of-64-bit-range integer fields.
Recommend: document the limitation.

### 8.3 Duplicate JSON keys collapse

`to_string` emits `{"a":1,"a":2}`; `to_value` produces `{"a":2}`.
Pathological, arguably a bug in the user's type. Noted, not a blocker.

### 8.4 `RawValue` is feature-gated off

This repo does not enable `serde_json/raw_value`. If a downstream crate does,
a payload containing `RawValue` would lose its exact original text through
`to_value`. Remote edge case; one line in trait docs.

### 8.5 Verified NOT to be problems

`f64::NAN/INFINITY` (→ null both ways), `-0.0`, `1e300`, `0.1+0.2`,
`u64::MAX`, `i128` within `u64` range, `HashMap<i32,_>`, tuples, `char`
(incl. non-BMP), `Vec<u8>`, `Option::None`.

### 8.6 No path genuinely needs pre-rendered text

All 17 sites checked. `callback` is deserialize-only (wire text comes from
external caller), which the proposal handles directly (`data: &str` input).
`invoke`'s pre-rendered input is a lifecycle constraint resolved in §1d.

## 9. Suggested sequencing (not part of this slice)

1. Spec amendment (owner): new trait shape; Value in the public API; the §7
   decision; the §8.2 limitation.
2. Trait + FileSystemSerdes (`src/serdes.rs`), retaining the `{"raw":...}`
   reader arm.
3. The 12 helpers + the invoke builder field (§1d, §2). Delete
   `serialize_item_value` / `deserialize_item_value`.
4. In-tree tests/doctests (§6d).
5. `compliance/` ×10 and `examples/` ×3 (§6b, §6c).
6. `make check`; then the conformance suite.
7. Follow-up: map/parallel `default_serdes` gap (§5).

## Sources

- `~/github/aws-durable-execution-sdk-go/durable/context.go:96-99` — accessed 2026-07-31
- `~/github/aws-durable-execution-sdk-js/packages/aws-durable-execution-sdk-js/src/utils/serdes/serdes.ts:23-44` — accessed 2026-07-31
- `~/github/aws-durable-execution-sdk-java/sdk/src/main/java/software/amazon/lambda/durable/serde/SerDes.java:19,37` — accessed 2026-07-31
- `serde_json-1.0.151/src/value/mod.rs:108-135` (registry source) — accessed 2026-07-31
- Scratch crates (outside the repo, nothing in-tree changed): `/tmp/serdes-objsafe-v1`, `/tmp/serdes-fidelity-v1` — 2026-07-31
