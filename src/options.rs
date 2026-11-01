//! Configuration types for the durable execution runtime.

use std::sync::Arc;
use std::time::Duration;

use crate::client::ExecutionClient;

/// Error returned when [`OptionsBuilder::build()`] detects an invalid
/// configuration combination.
///
/// This error fires at construction time, never mid-execution, matching
/// the spec's construction-time-only validation rule.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk::Options;
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
/// jittered retry for transient failures. To customize transport behavior,
/// retry policy, timeouts, endpoint, credentials, build the client (or an
/// `aws_config::SdkConfig`) yourself and supply it via
/// [`lambda_client`](OptionsBuilder::lambda_client) (or
/// [`sdk_config`](OptionsBuilder::sdk_config)); the SDK uses it as-is.
///
/// To replace the transport itself, supply an
/// [`ExecutionClient`](crate::ExecutionClient) via
/// [`execution_client`](OptionsBuilder::execution_client). The SDK then
/// sends both durable execution service calls through it instead of the
/// Lambda client. [`LambdaExecutionClient::new`](crate::LambdaExecutionClient::new)
/// builds the default transport, so a custom transport can wrap it.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk::Options;
///
/// let options = Options::builder()
///     .build()
///     .expect("valid default config");
/// # drop(options);
/// ```
#[derive(Clone)]
#[non_exhaustive]
pub struct Options {
    /// A user-built AWS SDK config used to construct the Lambda client.
    pub(crate) sdk_config: Option<aws_config::SdkConfig>,
    /// A pre-built Lambda client. When set, it is used directly instead of
    /// building one from [`Self::sdk_config`].
    pub(crate) lambda_client: Option<aws_sdk_lambda::Client>,
    /// A caller-supplied transport. When set, it replaces the Lambda
    /// transport for both service calls.
    pub(crate) execution_client: Option<Arc<dyn ExecutionClient>>,
    /// Coalescing window for checkpoint writes. `None` (the default) adds
    /// no window: with batching on (the default), writes batch with a zero
    /// window. Immediate writes need `checkpoint_batching(false)` without a
    /// delay.
    pub(crate) checkpoint_delay: Option<Duration>,
    /// Whether checkpoint writes batch behind the single-writer flush lock
    /// even without a coalescing window. `true` (the default) batches with
    /// a zero window. `false` without a delay writes each checkpoint
    /// immediately.
    pub(crate) checkpoint_batching: bool,
}

impl Default for Options {
    /// Creates default options suitable for standard Lambda deployments.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::Options;
    ///
    /// let opts = Options::default();
    /// # drop(opts);
    /// ```
    fn default() -> Self {
        Self {
            sdk_config: None,
            lambda_client: None,
            execution_client: None,
            checkpoint_delay: None,
            checkpoint_batching: true,
        }
    }
}

/// The `Debug` text of an optional transport: `None`, or a placeholder,
/// because [`ExecutionClient`] does not require `Debug`.
struct ExecutionClientDebug<'a>(&'a Option<Arc<dyn ExecutionClient>>);

impl std::fmt::Debug for ExecutionClientDebug<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(_) => f.write_str("<custom execution client>"),
            None => f.write_str("None"),
        }
    }
}

// Written by hand: `Arc<dyn ExecutionClient>` does not implement `Debug`.
impl std::fmt::Debug for Options {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Options")
            .field("sdk_config", &self.sdk_config)
            .field("lambda_client", &self.lambda_client)
            .field(
                "execution_client",
                &ExecutionClientDebug(&self.execution_client),
            )
            .field("checkpoint_delay", &self.checkpoint_delay)
            .field("checkpoint_batching", &self.checkpoint_batching)
            .finish()
    }
}

impl Options {
    /// Creates a new [`OptionsBuilder`].
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::Options;
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
/// use aws_durable_execution_sdk::{Options, OptionsBuilder};
///
/// let options = Options::builder()
///     .build()
///     .expect("valid config");
/// # drop(options);
/// ```
#[derive(Default)]
#[must_use = "builders do nothing unless .build() is called"]
#[non_exhaustive]
pub struct OptionsBuilder {
    sdk_config: Option<aws_config::SdkConfig>,
    lambda_client: Option<aws_sdk_lambda::Client>,
    execution_client: Option<Arc<dyn ExecutionClient>>,
    checkpoint_delay: Option<Duration>,
    checkpoint_batching: Option<bool>,
}

// Written by hand: `Arc<dyn ExecutionClient>` does not implement `Debug`.
impl std::fmt::Debug for OptionsBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OptionsBuilder")
            .field("sdk_config", &self.sdk_config)
            .field("lambda_client", &self.lambda_client)
            .field(
                "execution_client",
                &ExecutionClientDebug(&self.execution_client),
            )
            .field("checkpoint_delay", &self.checkpoint_delay)
            .field("checkpoint_batching", &self.checkpoint_batching)
            .finish()
    }
}

impl OptionsBuilder {
    /// Sets the AWS SDK configuration for building service clients.
    ///
    /// Use this to provide custom endpoint configuration, credentials, or
    /// HTTP client settings.
    ///
    /// By default, when neither this nor
    /// [`lambda_client`](Self::lambda_client) is set, the SDK loads the
    /// ambient environment configuration
    /// (`aws_config::load_defaults`) and builds a client from it. That
    /// default client uses a 5 s connect timeout, a 50 s read timeout and
    /// a 55 s attempt timeout. A client built from a config supplied here
    /// keeps the config's own timeouts, and the SDK adds none.
    ///
    /// The parameter type is re-exported at the crate root as
    /// [`SdkConfig`](crate::SdkConfig), so callers can name it without
    /// depending on `aws-config` directly.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk::Options;
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
    /// from the SDK config. By default, when neither this nor
    /// [`sdk_config`](Self::sdk_config) is set, the SDK builds a client
    /// from the ambient environment configuration
    /// (`aws_config::load_defaults`), with a 5 s connect timeout, a 50 s
    /// read timeout and a 55 s attempt timeout. A client supplied here
    /// keeps its own timeouts, and the SDK adds none.
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
    /// use aws_durable_execution_sdk::{LambdaClient, Options, SdkConfig};
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

    /// Sets the transport for the two durable execution service calls.
    ///
    /// The SDK sends every checkpoint and every state read through
    /// `client` instead of the Lambda transport. See the
    /// [`ExecutionClient`](crate::ExecutionClient) contract for what a
    /// transport returns and how each error scope ends the invocation or
    /// the execution. [`LambdaExecutionClient::new`](crate::LambdaExecutionClient::new)
    /// builds the default transport, so a custom transport can wrap it.
    ///
    /// The transport replaces the Lambda client, so it cannot be set
    /// together with [`lambda_client`](Self::lambda_client) or
    /// [`sdk_config`](Self::sdk_config): [`build`](Self::build) rejects
    /// that combination.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use aws_durable_execution_sdk::{LambdaClient, LambdaExecutionClient, Options, SdkConfig};
    ///
    /// let client = LambdaClient::new(&SdkConfig::builder().build());
    /// let opts = Options::builder()
    ///     .execution_client(Arc::new(LambdaExecutionClient::new(client)))
    ///     .build()
    ///     .expect("valid config");
    /// # drop(opts);
    /// ```
    pub fn execution_client(mut self, client: Arc<dyn ExecutionClient>) -> Self {
        self.execution_client = Some(client);
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
    /// - **Suspension**: before an invocation reports `PENDING` to the
    ///   service, every buffered checkpoint is written, so no recorded
    ///   progress is lost across the suspend/resume boundary.
    /// - **Execution completion**: before an invocation reports its
    ///   terminal `SUCCEEDED`/`FAILED` envelope, the buffer is drained.
    /// - **Callback creation**: creating a callback flushes immediately,
    ///   because the service assigns the callback ID in the checkpoint
    ///   response and the handler needs that ID right away.
    ///
    /// When unset (the default), no window is added. Batching, which is on
    /// by default, still never holds a write back.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::Options;
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
    /// unconditionally, and any in-flight batched write is awaited, at
    /// these points:
    ///
    /// - **Suspension**: before an invocation reports `PENDING` to the
    ///   service, every buffered checkpoint is written, so no recorded
    ///   progress is lost across the suspend/resume boundary.
    /// - **Execution completion**: before an invocation reports its
    ///   terminal `SUCCEEDED`/`FAILED` envelope, the buffer is drained.
    /// - **Callback creation**: creating a callback flushes immediately,
    ///   because the service assigns the callback ID in the checkpoint
    ///   response and the handler needs that ID right away.
    ///
    /// Batching is on by default. `false` sends each checkpoint in its own
    /// request as soon as it is written.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::Options;
    ///
    /// let opts = Options::builder()
    ///     .checkpoint_batching(false)
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
    /// - Setting `execution_client` together with `lambda_client` or
    ///   `sdk_config` is invalid (the execution client replaces the Lambda
    ///   transport: provide one or the other). This check runs first.
    /// - Setting both `sdk_config` and `lambda_client` is invalid (the
    ///   client supersedes the config: provide one or the other).
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk::Options;
    ///
    /// let opts = Options::builder().build();
    /// assert!(opts.is_ok());
    /// # drop(opts);
    /// ```
    pub fn build(self) -> Result<Options, OptionsValidationError> {
        // Construction-time-only validation: invalid combos error here,
        // not mid-execution. Matches the spec's "construction-time only" rule.
        if self.execution_client.is_some()
            && (self.lambda_client.is_some() || self.sdk_config.is_some())
        {
            return Err(OptionsValidationError {
                message: "cannot set `execution_client` together with `lambda_client` or \
                          `sdk_config`: the execution client replaces the Lambda transport; \
                          provide one or the other"
                    .to_owned(),
            });
        }
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
            execution_client: self.execution_client,
            checkpoint_delay: self.checkpoint_delay,
            checkpoint_batching: self.checkpoint_batching.unwrap_or(true),
        })
    }
}

#[cfg(test)]
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
    fn checkpoint_batching_defaults_on() {
        assert!(
            Options::default().checkpoint_batching,
            "Options::default() must batch checkpoints"
        );
        let opts = Options::builder().build().expect("valid");
        assert!(
            opts.checkpoint_batching,
            "unset batching must default to on in build()"
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

    /// A transport that implements no `Debug`, as a caller's may not.
    struct NoDebugTransport;

    impl ExecutionClient for NoDebugTransport {
        fn checkpoint(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            _updates: Vec<crate::lambda_types::OperationUpdate>,
        ) -> std::pin::Pin<
            Box<
                dyn Future<Output = Result<crate::CheckpointOutput, crate::ExecutionClientError>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(async {
                Ok(crate::CheckpointOutput::new(
                    Some("t".to_owned()),
                    vec![],
                    None,
                ))
            })
        }

        fn get_state(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            _marker: Option<&str>,
        ) -> std::pin::Pin<
            Box<
                dyn Future<Output = Result<crate::GetStateOutput, crate::ExecutionClientError>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(async { Ok(crate::GetStateOutput::new(vec![], None)) })
        }
    }

    fn transport() -> Arc<dyn ExecutionClient> {
        Arc::new(NoDebugTransport)
    }

    #[test]
    fn execution_client_alone_is_valid_and_is_stored() {
        let opts = Options::builder()
            .execution_client(transport())
            .build()
            .expect("execution_client alone is valid");
        assert!(
            opts.execution_client.is_some(),
            "execution_client must be preserved, not dropped"
        );
    }

    #[test]
    fn execution_client_and_lambda_client_conflict() {
        let err = Options::builder()
            .execution_client(transport())
            .lambda_client(dummy_lambda_client())
            .build()
            .expect_err("execution_client with lambda_client is rejected");
        let msg = err.to_string();
        assert!(msg.contains("`execution_client`"), "{msg}");
        assert!(msg.contains("`lambda_client`"), "{msg}");
    }

    #[test]
    fn execution_client_and_sdk_config_conflict() {
        let err = Options::builder()
            .execution_client(transport())
            .sdk_config(dummy_sdk_config())
            .build()
            .expect_err("execution_client with sdk_config is rejected");
        let msg = err.to_string();
        assert!(msg.contains("`execution_client`"), "{msg}");
        assert!(msg.contains("`sdk_config`"), "{msg}");
    }

    #[test]
    fn all_three_transports_report_the_execution_client_conflict() {
        let err = Options::builder()
            .execution_client(transport())
            .lambda_client(dummy_lambda_client())
            .sdk_config(dummy_sdk_config())
            .build()
            .expect_err("all three transports are rejected");
        assert_eq!(
            err.message,
            "cannot set `execution_client` together with `lambda_client` or `sdk_config`: \
             the execution client replaces the Lambda transport; provide one or the other"
        );
    }

    #[test]
    fn options_debug_prints_a_placeholder_for_the_execution_client() {
        let builder = Options::builder().execution_client(transport());
        let builder_debug = format!("{builder:?}");
        assert!(
            builder_debug.contains("<custom execution client>"),
            "{builder_debug}"
        );
        let opts = builder.build().expect("valid");
        let opts_debug = format!("{opts:?}");
        assert!(
            opts_debug.contains("<custom execution client>"),
            "{opts_debug}"
        );

        let unset_builder = format!("{:?}", Options::builder());
        assert!(
            unset_builder.contains("execution_client: None"),
            "{unset_builder}"
        );
        let unset_opts = format!("{:?}", Options::builder().build().expect("valid"));
        assert!(
            unset_opts.contains("execution_client: None"),
            "{unset_opts}"
        );
    }
}
