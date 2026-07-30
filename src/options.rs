//! Configuration types for the durable execution runtime.

use crate::Serdes;

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
    /// Execution-wide default serializer/deserializer, applied by a `step`,
    /// `run_in_child_context`, `invoke`, `callback`, or `wait_for_condition`
    /// operation that sets no serdes of its own. Per-operation `.serdes(...)`
    /// takes precedence, falling back to this default. Threaded into the root
    /// [`DurableContext`] by [`wrap`](crate::wrap).
    pub(crate) serdes: Option<Box<dyn Serdes>>,
    /// A user-built AWS SDK config used to construct the Lambda client.
    pub(crate) sdk_config: Option<aws_config::SdkConfig>,
    /// A pre-built Lambda client. When set, it is used directly instead of
    /// building one from [`Self::sdk_config`].
    pub(crate) lambda_client: Option<aws_sdk_lambda::Client>,
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
            serdes: None,
            sdk_config: None,
            lambda_client: None,
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
    serdes: Option<Box<dyn Serdes>>,
    sdk_config: Option<aws_config::SdkConfig>,
    lambda_client: Option<aws_sdk_lambda::Client>,
}

impl OptionsBuilder {
    /// Sets the execution-wide default serializer/deserializer.
    ///
    /// Applied by any `step`, `run_in_child_context`, `invoke`, `callback`,
    /// or `wait_for_condition` operation that sets no per-operation serdes of
    /// its own. Per-operation `.serdes(...)` on any builder takes precedence,
    /// falling back to this default. If not set, `serde_json` is used.
    /// `map`/`parallel` use their own per-operation item serdes and are not
    /// affected by this default.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust::{Options, Serdes};
    ///
    /// # struct MySerdes;
    /// # impl std::fmt::Debug for MySerdes {
    /// #     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { Ok(()) }
    /// # }
    /// # impl Serdes for MySerdes {
    /// #     fn serialize(&self, _: &dyn std::any::Any) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> { todo!() }
    /// #     fn deserialize_bytes(&self, _: &[u8], _: &str) -> Result<Box<dyn std::any::Any + Send>, Box<dyn std::error::Error + Send + Sync>> { todo!() }
    /// # }
    /// let opts = Options::builder()
    ///     .serdes(MySerdes)
    ///     .build()
    ///     .expect("valid config");
    /// ```
    pub fn serdes(mut self, serdes: impl Serdes + 'static) -> Self {
        self.serdes = Some(Box::new(serdes));
        self
    }

    /// Sets the AWS SDK configuration for building service clients.
    ///
    /// Use this to provide custom endpoint configuration, credentials, or
    /// HTTP client settings.
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
    pub fn sdk_config(mut self, config: aws_config::SdkConfig) -> Self {
        self.sdk_config = Some(config);
        self
    }

    /// Sets a pre-built Lambda client.
    ///
    /// When provided, the SDK uses this client instead of building one
    /// from the SDK config.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust::Options;
    ///
    /// # async fn example() {
    /// let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    /// let client = aws_sdk_lambda::Client::new(&config);
    /// let opts = Options::builder()
    ///     .lambda_client(client)
    ///     .build()
    ///     .expect("valid config");
    /// # drop(opts);
    /// # }
    /// ```
    pub fn lambda_client(mut self, client: aws_sdk_lambda::Client) -> Self {
        self.lambda_client = Some(client);
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
            serdes: self.serdes,
            sdk_config: self.sdk_config,
            lambda_client: self.lambda_client,
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
}
