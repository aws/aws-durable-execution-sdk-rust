//! The plugin's configuration: when to emit, which operations to include,
//! and which payloads a record carries.

use std::fmt;
use std::sync::Arc;

use aws_durable_execution_sdk::BoxError;
use serde_json::Value;

use crate::contain::{contained, describe_error, log_contained};

/// When the plugin emits a record.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum EmitMode {
    /// One record when an invocation completes or fails the execution.
    /// This is the default.
    #[default]
    OnComplete,
    /// A `RUNNING` record when an invocation starts and on every
    /// checkpoint response that changes an operation, and a record at
    /// every invocation end, terminal or not. The invocation-start
    /// record is exported in the background, so the handler does not wait
    /// for it. A newer record that is ready first replaces it.
    OnChange,
    /// One record when an invocation fails the execution.
    OnFailure,
}

/// Which operations a record lists.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum OperationDetail {
    /// Only operations without a parent. This is the default.
    ///
    /// The SDK may omit the children of a context that completed in an
    /// earlier invocation. A top-level operation is always present. So the
    /// top-level list is the same whichever invocation builds it.
    #[default]
    TopLevel,
    /// Every named operation, including map items, parallel branches and
    /// the operations inside child contexts.
    ///
    /// The children of a context that completed in an earlier invocation
    /// are listed only when the SDK reports them. Raise
    /// `OptionsBuilder::plugin_child_operations_depth` to keep them.
    FullTree,
}

/// A transform of a JSON payload.
///
/// The transform receives the payload as JSON and returns the JSON the
/// record carries. An `Err` or a panic omits the payload from the record,
/// so a failing redaction never writes the original value.
pub type Transform = Arc<dyn Fn(Value) -> Result<Value, BoxError> + Send + Sync>;

/// Applies a transform. Returns `None` when the transform returns an error
/// or panics.
///
/// The error that a transform returns is the transform's value. Formatting
/// it and dropping it run the transform author's code, and either can
/// panic. So the catch covers the call only, and `describe_error` formats
/// the error and drops it in two separate catches.
pub(crate) fn apply_transform(transform: &Transform, value: Value) -> Option<Value> {
    match contained(|| transform(value)) {
        Ok(Ok(value)) => Some(value),
        Ok(Err(error)) => {
            let error = describe_error(error);
            log_contained(|| {
                tracing::debug!(target: crate::TARGET, %error, "telemetry transform failed; omitting the value");
            });
            None
        }
        Err(_) => {
            log_contained(|| {
                tracing::debug!(target: crate::TARGET, "telemetry transform panicked; omitting the value");
            });
            None
        }
    }
}

/// Whether a record carries the execution input or output, and in which
/// form.
#[derive(Clone, Default)]
#[non_exhaustive]
pub enum ContentSetting {
    /// The record carries the payload as the execution received or
    /// returned it. This is the default.
    #[default]
    Include,
    /// The record omits the payload.
    Omit,
    /// The record carries the transform's result. See [`Transform`].
    Transform(Transform),
}

impl ContentSetting {
    /// A setting that applies `transform` to the payload.
    pub fn transform<F>(transform: F) -> Self
    where
        F: Fn(Value) -> Result<Value, BoxError> + Send + Sync + 'static,
    {
        Self::Transform(Arc::new(transform))
    }

    /// The value the record carries for `value` under this setting.
    pub(crate) fn apply(&self, value: Option<Value>) -> Option<Value> {
        match self {
            Self::Omit => None,
            Self::Include => value,
            Self::Transform(transform) => apply_transform(transform, value?),
        }
    }
}

impl fmt::Debug for ContentSetting {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Include => f.write_str("Include"),
            Self::Omit => f.write_str("Omit"),
            Self::Transform(_) => f.write_str("Transform(..)"),
        }
    }
}

/// The record settings of every operation with one name.
///
/// By default a record omits every operation result. An override with a
/// [`result`](Self::result) transform includes the result of the named
/// operation in the form the transform returns.
#[derive(Clone)]
#[non_exhaustive]
pub struct OperationOverride {
    operation_name: String,
    exclude: bool,
    result: Option<Transform>,
}

impl OperationOverride {
    /// An override for the operations named `operation_name`. It changes
    /// nothing until a setter adds a setting.
    pub fn new(operation_name: impl Into<String>) -> Self {
        Self {
            operation_name: operation_name.into(),
            exclude: false,
            result: None,
        }
    }

    /// Omits the named operations from the record.
    #[must_use]
    pub fn exclude(mut self) -> Self {
        self.exclude = true;
        self
    }

    /// Includes the result of the named operations, transformed.
    ///
    /// The transform receives the checkpointed result parsed as JSON. A
    /// result that is not valid JSON reaches it as a JSON string. An `Err`
    /// or a panic omits the result.
    #[must_use]
    pub fn result<F>(mut self, transform: F) -> Self
    where
        F: Fn(Value) -> Result<Value, BoxError> + Send + Sync + 'static,
    {
        self.result = Some(Arc::new(transform));
        self
    }

    /// The operation name this override applies to.
    pub fn operation_name(&self) -> &str {
        &self.operation_name
    }

    pub(crate) fn is_excluded(&self) -> bool {
        self.exclude
    }

    pub(crate) fn result_transform(&self) -> Option<&Transform> {
        self.result.as_ref()
    }
}

impl fmt::Debug for OperationOverride {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OperationOverride")
            .field("operation_name", &self.operation_name)
            .field("exclude", &self.exclude)
            .field("result", &self.result.as_ref().map(|_| ".."))
            .finish()
    }
}

/// Which payloads a record carries.
///
/// The default carries the execution input and output and every
/// operation error, and omits every operation result.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ContentConfig {
    pub(crate) input: ContentSetting,
    pub(crate) output: ContentSetting,
    pub(crate) overrides: Vec<OperationOverride>,
    pub(crate) include_errors: bool,
}

impl Default for ContentConfig {
    fn default() -> Self {
        Self {
            input: ContentSetting::Include,
            output: ContentSetting::Include,
            overrides: Vec::new(),
            include_errors: true,
        }
    }
}

impl ContentConfig {
    /// The default content: input, output and errors, no results.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets how the record carries the execution input.
    #[must_use]
    pub fn input(mut self, setting: ContentSetting) -> Self {
        self.input = setting;
        self
    }

    /// Sets how the record carries the execution output.
    #[must_use]
    pub fn output(mut self, setting: ContentSetting) -> Self {
        self.output = setting;
        self
    }

    /// Adds an operation override. When two overrides name the same
    /// operation, the later one applies.
    #[must_use]
    pub fn operation_override(mut self, operation_override: OperationOverride) -> Self {
        self.overrides.push(operation_override);
        self
    }

    /// Sets whether operation records carry their errors. The default is
    /// `true`. The execution error is always carried.
    #[must_use]
    pub fn include_errors(mut self, include: bool) -> Self {
        self.include_errors = include;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contain::tests::{
        PanicsOnDisplay, PanicsOnDisplayAndDrop, PanicsOnDrop, PanicsOnEvent,
    };
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn include_keeps_the_value() {
        assert_eq!(
            ContentSetting::Include.apply(Some(json!({"a": 1}))),
            Some(json!({"a": 1}))
        );
    }

    #[test]
    fn omit_drops_the_value() {
        assert_eq!(ContentSetting::Omit.apply(Some(json!(1))), None);
    }

    #[test]
    fn transform_replaces_the_value() {
        let setting = ContentSetting::transform(|value| Ok(json!({"wrapped": value})));
        assert_eq!(setting.apply(Some(json!(1))), Some(json!({"wrapped": 1})));
    }

    #[test]
    fn transform_is_not_called_without_a_value() {
        let setting = ContentSetting::transform(|_| Ok(json!("called")));
        assert_eq!(setting.apply(None), None);
    }

    #[test]
    fn a_failing_transform_omits_the_value() {
        let setting = ContentSetting::transform(|_| Err("redaction failed".into()));
        assert_eq!(setting.apply(Some(json!("secret"))), None);
    }

    #[test]
    #[expect(
        clippy::panic,
        reason = "the test checks that a panicking transform is contained"
    )]
    fn a_panicking_transform_omits_the_value() {
        let setting = ContentSetting::transform(|_| panic!("redaction panicked"));
        assert_eq!(setting.apply(Some(json!("secret"))), None);
    }

    #[test]
    fn an_error_whose_formatting_panics_omits_the_value() {
        // The subscriber formats the logged error, which runs its `Display`.
        let _subscriber = test_tracing::install_test_subscriber(test_tracing::FormatsFields);
        let setting = ContentSetting::transform(|_| Err(PanicsOnDisplay.into()));
        assert_eq!(setting.apply(Some(json!("secret"))), None);
    }

    #[test]
    fn an_error_whose_drop_panics_omits_the_value() {
        let setting = ContentSetting::transform(|_| Err(PanicsOnDrop.into()));
        assert_eq!(setting.apply(Some(json!("secret"))), None);
    }

    #[test]
    #[expect(
        clippy::panic,
        reason = "the test checks that a panicking payload drop is contained"
    )]
    fn a_panic_payload_whose_drop_panics_omits_the_value() {
        let setting = ContentSetting::transform(|_| std::panic::panic_any(PanicsOnDrop));
        assert_eq!(setting.apply(Some(json!("secret"))), None);
    }

    #[test]
    fn an_error_whose_formatting_and_drop_panic_omits_the_value() {
        // The subscriber formats the logged error. So code that passes the
        // error itself to the log also runs its `Display`.
        let _subscriber = test_tracing::install_test_subscriber(test_tracing::FormatsFields);
        let setting = ContentSetting::transform(|_| Err(PanicsOnDisplayAndDrop.into()));
        assert_eq!(setting.apply(Some(json!("secret"))), None);
    }

    #[test]
    #[expect(
        clippy::panic,
        reason = "the test checks that a panicking transform is contained"
    )]
    fn a_subscriber_that_panics_on_the_log_of_a_panicking_transform_omits_the_value() {
        let events = Arc::new(AtomicUsize::new(0));
        let _subscriber = test_tracing::install_test_subscriber(PanicsOnEvent(events.clone()));
        let setting = ContentSetting::transform(|_| panic!("redaction panicked"));
        assert_eq!(setting.apply(Some(json!("secret"))), None);
        assert_eq!(events.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn the_default_content_includes_payloads_and_errors() {
        let content = ContentConfig::new();
        assert!(matches!(content.input, ContentSetting::Include));
        assert!(matches!(content.output, ContentSetting::Include));
        assert!(content.include_errors);
        assert!(content.overrides.is_empty());
    }

    #[test]
    fn setters_replace_the_defaults() {
        let content = ContentConfig::new()
            .input(ContentSetting::Omit)
            .output(ContentSetting::Omit)
            .include_errors(false)
            .operation_override(OperationOverride::new("a").exclude());
        assert!(matches!(content.input, ContentSetting::Omit));
        assert!(matches!(content.output, ContentSetting::Omit));
        assert!(!content.include_errors);
        assert_eq!(content.overrides.len(), 1);
    }

    #[test]
    fn an_override_records_its_settings() {
        let plain = OperationOverride::new("fetch");
        assert_eq!(plain.operation_name(), "fetch");
        assert!(!plain.is_excluded());
        assert!(plain.result_transform().is_none());

        let full = OperationOverride::new("fetch").exclude().result(Ok);
        assert!(full.is_excluded());
        assert!(full.result_transform().is_some());
    }

    #[test]
    fn debug_hides_the_transforms() {
        let setting = ContentSetting::transform(Ok);
        assert_eq!(format!("{setting:?}"), "Transform(..)");
        assert_eq!(format!("{:?}", ContentSetting::Include), "Include");
        assert_eq!(format!("{:?}", ContentSetting::Omit), "Omit");
        let debug = format!("{:?}", OperationOverride::new("x").result(Ok));
        assert!(debug.contains("result: Some(\"..\")"), "{debug}");
    }
}
