//! Configuration types for the durable execution runtime.

use std::time::Duration;

/// Error returned when [`OptionsBuilder::build()`] detects an invalid
/// configuration combination.
///
/// This error fires at construction time — never mid-execution — matching
/// the spec's construction-time-only validation rule.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk_rust::Options;
///
/// // Attempting to set both sdk_config and lambda_client errors at build():
/// // let opts = Options::builder()
/// //     .sdk_config(config)
/// //     .lambda_client(client)
/// //     .build(); // Err(OptionsValidationError { ... })
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OptionsValidationError {
    message: String,
}

impl std::fmt::Display for OptionsValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid Options configuration: {}", self.message)
    }
}

impl std::error::Error for OptionsValidationError {}

/// Configuration for the durable execution runtime.
///
/// Use [`Options::builder()`] to construct. All settings are optional;
/// defaults are suitable for standard Lambda deployments.
///
/// # Customizing the service client
///
/// The SDK calls the durable execution service through an
/// `aws_sdk_lambda::Client` and relies on that client's own standard,
/// jittered retry for transient failures. To customize transport behavior —
/// retry policy, timeouts, endpoint, credentials — build the client (or an
/// `aws_config::SdkConfig`) yourself and supply it via
/// [`lambda_client`](OptionsBuilder::lambda_client) (or
/// [`sdk_config`](OptionsBuilder::sdk_config)); the SDK uses it as-is.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk_rust::Options;
///
/// let options = Options::builder()
///     .build()
///     .expect("valid default config");
/// # drop(options);
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct Options {
    /// A user-built AWS SDK config used to construct the Lambda client.
    pub(crate) sdk_config: Option<aws_config::SdkConfig>,
    /// A pre-built Lambda client. When set, it is used directly instead of
    /// building one from [`Self::sdk_config`].
    pub(crate) lambda_client: Option<aws_sdk_lambda::Client>,
    /// Coalescing window for checkpoint writes. `None` (the default) writes
    /// every checkpoint immediately, exactly as before this knob existed.
    pub(crate) checkpoint_delay: Option<Duration>,
    /// Whether checkpoint writes batch behind the single-writer flush lock
    /// even without a coalescing window. `false` (the default) keeps
    /// immediate writes.
    pub(crate) checkpoint_batching: bool,
}

impl Default for Options {
    /// Creates default options suitable for standard Lambda deployments.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::Options;
    ///
    /// let opts = Options::default();
    /// # drop(opts);
    /// ```
    fn default() -> Self {
        Self {
            sdk_config: None,
            lambda_client: None,
            checkpoint_delay: None,
            checkpoint_batching: false,
        }
    }
}

impl Options {
    /// Creates a new [`OptionsBuilder`].
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::Options;
    ///
    /// let builder = Options::builder();
    /// let opts = builder.build().expect("valid config");
    /// # drop(opts);
    /// ```
    pub fn builder() -> OptionsBuilder {
        OptionsBuilder::default()
    }
}

/// Builder for [`Options`].
///
/// Follows the Rust API Guidelines C-BUILDER pattern. All methods consume
/// and return `self` for chaining.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk_rust::{Options, OptionsBuilder};
///
/// let options = Options::builder()
///     .build()
///     .expect("valid config");
/// # drop(options);
/// ```
#[derive(Debug, Default)]
#[must_use = "builders do nothing unless .build() is called"]
#[non_exhaustive]
pub struct OptionsBuilder {
    sdk_config: Option<aws_config::SdkConfig>,
    lambda_client: Option<aws_sdk_lambda::Client>,
    checkpoint_delay: Option<Duration>,
    checkpoint_batching: Option<bool>,
}

impl OptionsBuilder {
    /// Sets the AWS SDK configuration for building service clients.
    ///
    /// Use this to provide custom endpoint configuration, credentials, or
    /// HTTP client settings.
    ///
    /// The parameter type is re-exported at the crate root as
    /// [`SdkConfig`](crate::SdkConfig), so callers can name it without
    /// depending on `aws-config` directly.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust::Options;
    ///
    /// # async fn example() {
    /// let sdk_config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    /// let opts = Options::builder()
    ///     .sdk_config(sdk_config)
    ///     .build()
    ///     .expect("valid config");
    /// # drop(opts);
    /// # }
    /// ```
    pub fn sdk_config(mut self, config: crate::SdkConfig) -> Self {
        self.sdk_config = Some(config);
        self
    }

    /// Sets a pre-built Lambda client.
    ///
    /// When provided, the SDK uses this client instead of building one
    /// from the SDK config.
    ///
    /// This is the supported path for customizing how the SDK talks to the
    /// durable execution service: configure retry, timeouts, endpoint, or
    /// credentials on the client (or use
    /// [`sdk_config`](Self::sdk_config)) and the SDK uses it verbatim.
    /// Transient-failure retry of the service calls is the client's own
    /// standard retry; the SDK adds no retry layer of its own.
    ///
    /// The parameter type is re-exported at the crate root as
    /// [`LambdaClient`](crate::LambdaClient), so callers can name it without
    /// depending on `aws-sdk-lambda` directly.
    ///
    /// # Examples
    ///
    /// The client and its configuration type are both reachable through the
    /// crate root, so this compiles with no direct `aws-config` or
    /// `aws-sdk-lambda` dependency:
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust::{LambdaClient, Options, SdkConfig};
    ///
    /// let config = SdkConfig::builder().build();
    /// let client = LambdaClient::new(&config);
    /// let opts = Options::builder()
    ///     .lambda_client(client)
    ///     .build()
    ///     .expect("valid config");
    /// # drop(opts);
    /// ```
    pub fn lambda_client(mut self, client: crate::LambdaClient) -> Self {
        self.lambda_client = Some(client);
        self
    }

    /// Defers checkpoint writes for up to `delay` so that checkpoints from
    /// concurrently running operations coalesce into fewer service calls.
    ///
    /// With a delay configured, a checkpoint written while other operations
    /// are also checkpointing joins a shared batch; the batch is sent as a
    /// single `CheckpointDurableExecution` call when the window elapses.
    /// For high-fan-out executions (large `map`/`parallel` batches) this
    /// trades up to `delay` of extra latency per checkpoint for
    /// substantially fewer API calls. A coalesced batch is split to respect
    /// the service's per-request limits (operation count and payload size),
    /// preserving write order across the splits. Every operation still
    /// awaits its own checkpoint before proceeding, so ordering and replay
    /// semantics are unchanged.
    ///
    /// # Flush contract
    ///
    /// A checkpoint that must land before the execution can make progress
    /// is never held back by the window. Pending checkpoints are flushed
    /// unconditionally at these points:
    ///
    /// - **Suspension** — before an invocation reports `PENDING` to the
    ///   service, every buffered checkpoint is written, so no recorded
    ///   progress is lost across the suspend/resume boundary.
    /// - **Execution completion** — before an invocation reports its
    ///   terminal `SUCCEEDED`/`FAILED` envelope, the buffer is drained.
    /// - **Callback creation** — creating a callback flushes immediately,
    ///   because the service assigns the callback ID in the checkpoint
    ///   response and the handler needs that ID right away.
    ///
    /// When unset (the default), every checkpoint is written immediately.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::Options;
    /// use std::time::Duration;
    ///
    /// let opts = Options::builder()
    ///     .checkpoint_delay(Duration::from_millis(20))
    ///     .build()
    ///     .expect("valid config");
    /// # drop(opts);
    /// ```
    pub fn checkpoint_delay(mut self, delay: Duration) -> Self {
        self.checkpoint_delay = Some(delay);
        self
    }

    /// Batches multiple checkpoint requests into fewer API calls, without
    /// adding latency to any individual checkpoint.
    ///
    /// With batching enabled, checkpoint writes go through a single ordered
    /// writer. A checkpoint that arrives while an earlier write is still in
    /// flight joins a shared buffer, and the whole buffer is sent together
    /// in the next `CheckpointDurableExecution` call once the writer is
    /// free. Under high fan-out (large `map`/`parallel` batches) this
    /// substantially reduces API calls; a sequential handler sees one call
    /// per checkpoint exactly as today. Batches are split to respect the
    /// service's per-request limits (operation count and payload size),
    /// preserving write order across the splits, and every operation still
    /// awaits its own checkpoint before proceeding, so ordering and replay
    /// semantics are unchanged.
    ///
    /// Combine with [`checkpoint_delay`](Self::checkpoint_delay) to also
    /// hold writes open for a coalescing window; batching alone never
    /// delays a write.
    ///
    /// # Flush contract
    ///
    /// A checkpoint that must land before the execution can make progress
    /// is never held back by batching. Pending checkpoints are flushed
    /// unconditionally — and any in-flight batched write is awaited — at
    /// these points:
    ///
    /// - **Suspension** — before an invocation reports `PENDING` to the
    ///   service, every buffered checkpoint is written, so no recorded
    ///   progress is lost across the suspend/resume boundary.
    /// - **Execution completion** — before an invocation reports its
    ///   terminal `SUCCEEDED`/`FAILED` envelope, the buffer is drained.
    /// - **Callback creation** — creating a callback flushes immediately,
    ///   because the service assigns the callback ID in the checkpoint
    ///   response and the handler needs that ID right away.
    ///
    /// When unset or `false` (the default), every checkpoint is written
    /// immediately.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::Options;
    ///
    /// let opts = Options::builder()
    ///     .checkpoint_batching(true)
    ///     .build()
    ///     .expect("valid config");
    /// # drop(opts);
    /// ```
    pub fn checkpoint_batching(mut self, enabled: bool) -> Self {
        self.checkpoint_batching = Some(enabled);
        self
    }

    /// Builds the [`Options`] from the configured values.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration contains invalid combinations:
    /// - Setting both `sdk_config` and `lambda_client` is invalid (the
    ///   client supersedes the config — provide one or the other).
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::Options;
    ///
    /// let opts = Options::builder().build();
    /// assert!(opts.is_ok());
    /// # drop(opts);
    /// ```
    pub fn build(self) -> Result<Options, OptionsValidationError> {
        // Construction-time-only validation: invalid combos error here,
        // not mid-execution. Matches the spec's "construction-time only" rule.
        if self.sdk_config.is_some() && self.lambda_client.is_some() {
            return Err(OptionsValidationError {
                message: "cannot set both `sdk_config` and `lambda_client` — \
                          the pre-built client supersedes the config; provide one or the other"
                    .to_owned(),
            });
        }

        Ok(Options {
            sdk_config: self.sdk_config,
            lambda_client: self.lambda_client,
            checkpoint_delay: self.checkpoint_delay,
            checkpoint_batching: self.checkpoint_batching.unwrap_or(false),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // reason: test assertions
#[allow(clippy::expect_used)] // reason: test assertions
mod tests {
    use super::*;

    /// An offline `SdkConfig` (no network, no credentials probe).
    fn dummy_sdk_config() -> aws_config::SdkConfig {
        aws_config::SdkConfig::builder().build()
    }

    /// An offline Lambda client (built from a static config; never calls AWS).
    fn dummy_lambda_client() -> aws_sdk_lambda::Client {
        let conf = aws_sdk_lambda::config::Config::builder()
            .behavior_version(aws_sdk_lambda::config::BehaviorVersion::latest())
            .region(aws_sdk_lambda::config::Region::new("us-east-1"))
            .build();
        aws_sdk_lambda::Client::from_conf(conf)
    }

    #[test]
    fn default_options_build_succeeds() {
        let result = Options::builder().build();
        assert!(result.is_ok());
    }

    #[test]
    #[allow(clippy::unwrap_used)] // reason: test assertion — extracting expected error
    fn sdk_config_and_lambda_client_conflict() {
        let result = Options::builder()
            .sdk_config(dummy_sdk_config())
            .lambda_client(dummy_lambda_client())
            .build();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("sdk_config"),
            "Error should mention sdk_config: {err_msg}"
        );
        assert!(
            err_msg.contains("lambda_client"),
            "Error should mention lambda_client: {err_msg}"
        );
    }

    #[test]
    fn sdk_config_alone_is_valid_and_is_stored() {
        let opts = Options::builder().sdk_config(dummy_sdk_config()).build();
        assert!(opts.is_ok());
        assert!(
            opts.expect("valid").sdk_config.is_some(),
            "sdk_config must be preserved, not dropped"
        );
    }

    #[test]
    fn lambda_client_alone_is_valid_and_is_stored() {
        let opts = Options::builder()
            .lambda_client(dummy_lambda_client())
            .build();
        assert!(opts.is_ok());
        assert!(
            opts.expect("valid").lambda_client.is_some(),
            "lambda_client must be preserved, not dropped"
        );
    }

    #[test]
    fn validation_error_implements_std_error() {
        let err = OptionsValidationError {
            message: "test".to_owned(),
        };
        // Verify it implements std::error::Error via trait object coercion.
        let _: &dyn std::error::Error = &err;
        assert!(err.to_string().contains("test"));
    }

    #[test]
    fn default_options_carry_no_execution_tuning() {
        let opts = Options::builder().build().expect("valid");
        assert!(
            opts.checkpoint_delay.is_none(),
            "unset checkpoint delay must stay None so checkpoints write immediately"
        );
    }

    #[test]
    fn checkpoint_delay_is_stored() {
        let opts = Options::builder()
            .checkpoint_delay(Duration::from_millis(20))
            .build()
            .expect("valid");
        assert_eq!(opts.checkpoint_delay, Some(Duration::from_millis(20)));
    }

    #[test]
    fn checkpoint_batching_enabled_builds_and_is_stored() {
        let opts = Options::builder()
            .checkpoint_batching(true)
            .build()
            .expect("checkpoint_batching(true) is a valid configuration");
        assert!(
            opts.checkpoint_batching,
            "the enabled batching knob must be preserved, not dropped"
        );
    }

    #[test]
    fn checkpoint_batching_disabled_is_valid() {
        let opts = Options::builder()
            .checkpoint_batching(false)
            .build()
            .expect("explicitly-disabled batching is valid");
        assert!(!opts.checkpoint_batching, "disabled batching stays off");
    }

    #[test]
    fn checkpoint_batching_defaults_off() {
        let opts = Options::builder().build().expect("valid");
        assert!(
            !opts.checkpoint_batching,
            "unset batching must stay off so checkpoints write immediately"
        );
    }

    #[test]
    fn checkpoint_batching_combines_with_delay() {
        let opts = Options::builder()
            .checkpoint_delay(Duration::from_millis(10))
            .checkpoint_batching(true)
            .build()
            .expect("delay and batching combine");
        assert_eq!(opts.checkpoint_delay, Some(Duration::from_millis(10)));
        assert!(opts.checkpoint_batching);
    }
}
