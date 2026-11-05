//! A recording plugin for the call-site tests of the user-function hooks.
//!
//! [`UfHooks::plugins`] registers two plugins, `p0` and `p1`, over one
//! log. Each records every operation, attempt and user-function hook call
//! and each enter and exit of its `wrap_user_function` (the records S, W
//! and E of the issue's fixtures), with what its probe returns then.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::{
    AttemptEndInfo, AttemptInfo, HookWork, InvocationInfo, InvocationPlugins, OperationInfo,
    Plugin, PluginFactory,
};
use crate::error::WireError;
use crate::{BoxError, OperationStatus};

/// One hook or wrap call that [`UfHooks`] recorded.
#[derive(Clone, Debug)]
pub(crate) struct UfRecord {
    /// `p0` or `p1`.
    pub(crate) plugin: &'static str,
    /// `start`, `attempt_start`, `uf_start`, `wrap_enter`, `wrap_exit`,
    /// `uf_end`, `attempt_end` or `end`.
    pub(crate) hook: &'static str,
    pub(crate) name: Option<String>,
    /// The info's `id()`.
    pub(crate) id: String,
    /// The info's `sub_type()`.
    pub(crate) sub_type: Option<String>,
    /// The info's `parent_id()`.
    pub(crate) parent_id: Option<String>,
    /// The info's `is_virtual()`.
    pub(crate) is_virtual: bool,
    pub(crate) replay: bool,
    pub(crate) status: Option<OperationStatus>,
    /// The info's `attempt()`.
    pub(crate) info_attempt: Option<u32>,
    /// An attempt hook's attempt number.
    pub(crate) attempt: Option<u32>,
    pub(crate) has_start_timestamp: bool,
    /// The info's `result()`.
    pub(crate) result: Option<String>,
    /// `uf_end`: whether the outcome is `Ok`.
    pub(crate) ok: Option<bool>,
    /// `uf_end`: the outcome's wire; `attempt_end`: its error.
    pub(crate) wire: Option<WireError>,
    /// The address of the info, for `std::ptr::eq` checks.
    pub(crate) info_addr: usize,
    /// The info's `Debug` form, for equality checks.
    pub(crate) info: String,
    /// What the probe returned when the hook ran.
    pub(crate) probe: String,
}

type Probe = dyn Fn() -> String + Send + Sync;

/// Whether `hook` is a user-function hook or wrap record.
fn is_user_function(hook: &str) -> bool {
    matches!(hook, "uf_start" | "wrap_enter" | "wrap_exit" | "uf_end")
}

/// A root context over `records` whose transport is a fresh in-memory
/// client, with `hooks`' `p0` and `p1` as the invocation's plugins and no
/// checkpoint buffer window.
pub(crate) fn context(
    hooks: &UfHooks,
    records: Vec<(String, crate::engine::CheckpointRecord)>,
) -> (
    crate::DurableContext,
    Arc<crate::client::InMemoryExecutionClient>,
) {
    let client = Arc::new(crate::client::InMemoryExecutionClient::new(Vec::new()));
    let ctx = context_over(
        hooks,
        records,
        Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
    );
    (ctx, client)
}

/// A root context over `records` and `client`, with `hooks`' `p0` and
/// `p1` as the invocation's plugins and no checkpoint buffer window.
pub(crate) fn context_over(
    hooks: &UfHooks,
    records: Vec<(String, crate::engine::CheckpointRecord)>,
    client: Arc<dyn crate::client::ExecutionClient>,
) -> crate::DurableContext {
    crate::DurableContext::new_root_with_client_and_defaults(
        "arn:test".to_owned(),
        lambda_runtime::Context::default(),
        Arc::new(crate::engine::CheckpointLog::from_records(records)),
        client,
        "token0".to_owned(),
        None,
        0,
        Some(hooks.plugins()),
    )
}

/// The first update named `name` with `action` that `updates` holds.
pub(crate) fn update_of(
    updates: &[aws_sdk_lambda::types::OperationUpdate],
    name: &str,
    action: &aws_sdk_lambda::types::OperationAction,
) -> Option<aws_sdk_lambda::types::OperationUpdate> {
    updates
        .iter()
        .find(|u| u.name() == Some(name) && u.action() == action)
        .cloned()
}

/// `(type, message)` of a wire error.
pub(crate) fn type_and_message(wire: &WireError) -> (Option<String>, Option<String>) {
    (
        wire.error_type().map(str::to_owned),
        wire.error_message().map(str::to_owned),
    )
}

/// `(type, message)` of the error object of `update`.
pub(crate) fn update_error(
    update: &aws_sdk_lambda::types::OperationUpdate,
) -> (Option<String>, Option<String>) {
    let error = update.error().expect("the update carries an error object");
    (
        error.error_type().map(str::to_owned),
        error.error_message().map(str::to_owned),
    )
}

/// The recording plugin. Clones share the log.
#[derive(Clone)]
pub(crate) struct UfHooks {
    tag: &'static str,
    records: Arc<Mutex<Vec<UfRecord>>>,
    probe: Option<Arc<Probe>>,
}

impl Default for UfHooks {
    fn default() -> Self {
        Self {
            tag: "p0",
            records: Arc::default(),
            probe: None,
        }
    }
}

impl UfHooks {
    /// A recorder whose records carry what `probe` returns at each call.
    pub(crate) fn probing(probe: impl Fn() -> String + Send + Sync + 'static) -> Self {
        Self {
            probe: Some(Arc::new(probe)),
            ..Self::default()
        }
    }

    /// The plugins of an invocation whose plugins are `p0` and `p1`, both
    /// recording into this log.
    pub(crate) fn plugins(&self) -> Arc<InvocationPlugins> {
        let factories: Vec<(&'static str, Arc<dyn PluginFactory>)> = ["p0", "p1"]
            .into_iter()
            .map(|tag| {
                let factory: Arc<dyn PluginFactory> = Arc::new(Self {
                    tag,
                    ..self.clone()
                });
                ("test::UfHooks", factory)
            })
            .collect();
        super::create_plugins(&factories, None, info).expect("two factories are registered")
    }

    /// Every record, in call order.
    pub(crate) fn records(&self) -> Vec<UfRecord> {
        self.records.lock().unwrap().clone()
    }

    /// `p0`'s records of the operation named `name`. Asserts that `p1`
    /// recorded the same hooks for it.
    pub(crate) fn of(&self, name: &str) -> Vec<UfRecord> {
        let records = self.records();
        let of = |tag: &str| -> Vec<UfRecord> {
            records
                .iter()
                .filter(|r| r.plugin == tag && r.name.as_deref() == Some(name))
                .cloned()
                .collect()
        };
        let p0 = of("p0");
        let hooks = |records: &[UfRecord]| records.iter().map(|r| r.hook).collect::<Vec<_>>();
        assert_eq!(hooks(&p0), hooks(&of("p1")), "p1's hooks of {name}");
        p0
    }

    /// The hooks of [`Self::of`].
    pub(crate) fn hooks_of(&self, name: &str) -> Vec<&'static str> {
        self.of(name).into_iter().map(|r| r.hook).collect()
    }

    /// `p0`'s first `hook` record of `name`.
    #[expect(clippy::panic, reason = "a missing record fails the test")]
    pub(crate) fn find(&self, name: &str, hook: &str) -> UfRecord {
        self.of(name)
            .into_iter()
            .find(|r| r.hook == hook)
            .unwrap_or_else(|| panic!("no {hook} record of {name}"))
    }

    /// `p0`'s user-function records of `name`: its start hook, a wrap enter
    /// and exit per poll, and its end hook when `ended`. Asserts that
    /// shape.
    pub(crate) fn user_function_of(&self, name: &str, ended: bool) -> Vec<UfRecord> {
        let records: Vec<UfRecord> = self
            .of(name)
            .into_iter()
            .filter(|r| is_user_function(r.hook))
            .collect();
        let hooks: Vec<&str> = records.iter().map(|r| r.hook).collect();
        let wraps_end = if ended {
            hooks.len().saturating_sub(1)
        } else {
            hooks.len()
        };
        let wraps = hooks.get(1..wraps_end).unwrap_or_default();
        assert_eq!(hooks.first(), Some(&"uf_start"), "{name}: {hooks:?}");
        assert_eq!(
            hooks.last() == Some(&"uf_end"),
            ended,
            "{name} ended: {hooks:?}"
        );
        assert!(
            !wraps.is_empty()
                && wraps
                    .chunks(2)
                    .all(|pair| pair == ["wrap_enter", "wrap_exit"]),
            "{name}: {hooks:?}"
        );
        records
    }

    /// Whether any user-function hook or wrap ran.
    pub(crate) fn any_user_function_record(&self) -> bool {
        self.records().iter().any(|r| is_user_function(r.hook))
    }

    fn push(
        &self,
        hook: &'static str,
        info: &OperationInfo,
        attempt: Option<u32>,
        outcome: Option<Result<(), &WireError>>,
        error: Option<&WireError>,
    ) {
        let probe = self.probe.as_ref().map(|probe| probe()).unwrap_or_default();
        let wire = match outcome {
            Some(Err(wire)) => Some(wire.clone()),
            _ => error.cloned(),
        };
        self.records.lock().unwrap().push(UfRecord {
            plugin: self.tag,
            hook,
            name: info.name().map(str::to_owned),
            id: info.id().to_owned(),
            sub_type: info.sub_type().map(str::to_owned),
            parent_id: info.parent_id().map(str::to_owned),
            is_virtual: info.is_virtual(),
            replay: info.is_replay(),
            status: info.status(),
            info_attempt: info.attempt(),
            attempt,
            has_start_timestamp: info.start_timestamp().is_some(),
            result: info.result().map(str::to_owned),
            ok: outcome.map(|o| o.is_ok()),
            wire,
            info_addr: std::ptr::from_ref(info) as usize,
            info: format!("{info:?}"),
            probe,
        });
    }
}

impl Plugin for UfHooks {
    fn name(&self) -> &'static str {
        self.tag
    }

    fn on_operation_start(&self, info: &OperationInfo) -> HookWork {
        self.push("start", info, None, None, None);
        Ok(None)
    }

    fn on_operation_end(&self, info: &OperationInfo) -> HookWork {
        self.push("end", info, None, None, None);
        Ok(None)
    }

    fn on_attempt_start(&self, info: &AttemptInfo) -> HookWork {
        self.push(
            "attempt_start",
            info.operation(),
            Some(info.attempt()),
            None,
            None,
        );
        Ok(None)
    }

    fn on_attempt_end(&self, info: &AttemptEndInfo) -> HookWork {
        self.push(
            "attempt_end",
            info.operation(),
            Some(info.attempt()),
            None,
            info.error(),
        );
        Ok(None)
    }

    fn wrap_user_function(&self, info: &OperationInfo, next: &mut dyn FnMut()) {
        self.push("wrap_enter", info, None, None, None);
        next();
        self.push("wrap_exit", info, None, None, None);
    }

    fn on_user_function_start(&self, info: &OperationInfo) -> HookWork {
        self.push("uf_start", info, None, None, None);
        Ok(None)
    }

    fn on_user_function_end(
        &self,
        info: &OperationInfo,
        outcome: Result<(), &WireError>,
    ) -> HookWork {
        self.push("uf_end", info, None, Some(outcome), None);
        Ok(None)
    }
}

impl PluginFactory for UfHooks {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(self.clone())))
    }
}

/// The info of a first invocation with no operations.
fn info() -> InvocationInfo {
    InvocationInfo::new(
        "r".to_owned(),
        "arn".to_owned(),
        "{}".to_owned(),
        std::collections::BTreeMap::new(),
        Vec::new(),
        None,
        true,
        None,
    )
}

type OnDisplay = dyn Fn() + Send + Sync;

/// Dc: an error whose `Display` counts its calls and runs `on_display`
/// at each.
#[derive(Clone)]
pub(crate) struct Counted {
    calls: Arc<AtomicUsize>,
    on_display: Option<Arc<OnDisplay>>,
}

impl Counted {
    pub(crate) fn new() -> Self {
        Self {
            calls: Arc::default(),
            on_display: None,
        }
    }

    /// A Dc that runs `on_display` at each `Display` call.
    pub(crate) fn calling(on_display: impl Fn() + Send + Sync + 'static) -> Self {
        Self {
            on_display: Some(Arc::new(on_display)),
            ..Self::new()
        }
    }

    /// A Dc that shares this one's count and runs `on_display` at each
    /// `Display` call.
    pub(crate) fn sharing(&self, on_display: impl Fn() + Send + Sync + 'static) -> Self {
        Self {
            calls: Arc::clone(&self.calls),
            on_display: Some(Arc::new(on_display)),
        }
    }

    /// The `Display` calls so far.
    pub(crate) fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    /// A boxed clone, as a body returns it.
    pub(crate) fn boxed(&self) -> BoxError {
        Box::new(self.clone())
    }
}

impl std::fmt::Debug for Counted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Counted")
    }
}

impl std::fmt::Display for Counted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(on_display) = &self.on_display {
            on_display();
        }
        f.write_str("counted")
    }
}

impl std::error::Error for Counted {}
