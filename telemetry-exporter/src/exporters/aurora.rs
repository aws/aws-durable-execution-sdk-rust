//! Writes one row per execution to an Aurora table through the RDS Data API.

use aws_durable_execution_sdk::BoxError;
use aws_sdk_rdsdata::Client;
use aws_sdk_rdsdata::error::{DisplayErrorContext, SdkError};
use aws_sdk_rdsdata::types::{Field, SqlParameter};

use crate::exporter::{ExportFuture, TelemetryExporter};
use crate::record::{ExecutionStatus, TelemetryRecord};
use crate::render::{OperationsFormat, render_json};

/// The default table name.
const DEFAULT_TABLE: &str = "durable_execution_telemetry";

/// The default record size limit in bytes.
const DEFAULT_MAX_RECORD_SIZE_BYTES: usize = 1_000_000;

/// The database engine of the Aurora cluster. It selects the SQL dialect of
/// the upsert statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AuroraEngine {
    /// Aurora PostgreSQL. The statement uses `ON CONFLICT ... DO UPDATE`.
    PostgreSql,
    /// Aurora MySQL. The statement uses `ON DUPLICATE KEY UPDATE`.
    MySql,
}

/// Writes each record as one row of an Aurora table.
///
/// The exporter sends one `ExecuteStatement` request of the RDS Data API
/// for each record. The statement inserts the row, or updates the existing
/// row of the same `execution_arn`. So the table holds the latest record of
/// each execution. The Data API needs no VPC and no connection pool. The
/// cluster must have the Data API enabled.
///
/// The table needs these columns, with `execution_arn` as the primary key
/// or a unique key:
/// `execution_arn`, `execution_name`, `function_name`, `status`,
/// `start_time`, `end_time`, `duration_ms`, `record_json` and `emitted_at`.
/// The exporter does not create the table. On PostgreSQL, the time columns
/// are `timestamptz` and `record_json` is `jsonb`. The statement casts
/// its text parameters to these types.
///
/// `record_json` holds the record with the `operations` array.
///
/// The function's role needs `rds-data:ExecuteStatement` on the cluster and
/// `secretsmanager:GetSecretValue` on the secret.
#[derive(Debug)]
pub struct AuroraExporter {
    resource_arn: String,
    secret_arn: String,
    database: String,
    table: String,
    engine: AuroraEngine,
    region: Option<String>,
    max_record_size_bytes: usize,
    client: tokio::sync::OnceCell<Client>,
}

/// Builds an [`AuroraExporter`].
#[derive(Debug)]
#[must_use]
pub struct AuroraExporterBuilder {
    resource_arn: String,
    secret_arn: String,
    database: String,
    table: String,
    engine: AuroraEngine,
    region: Option<String>,
    max_record_size_bytes: usize,
    client: Option<Client>,
}

impl AuroraExporter {
    /// A builder for an exporter that writes to the database `database` of
    /// the cluster `resource_arn`. The Data API reads the credentials from
    /// the Secrets Manager secret `secret_arn`. `engine` selects the SQL
    /// dialect.
    pub fn builder(
        resource_arn: impl Into<String>,
        secret_arn: impl Into<String>,
        database: impl Into<String>,
        engine: AuroraEngine,
    ) -> AuroraExporterBuilder {
        AuroraExporterBuilder {
            resource_arn: resource_arn.into(),
            secret_arn: secret_arn.into(),
            database: database.into(),
            table: DEFAULT_TABLE.to_owned(),
            engine,
            region: None,
            max_record_size_bytes: DEFAULT_MAX_RECORD_SIZE_BYTES,
            client: None,
        }
    }

    /// The client: the configured one, or one built from the default
    /// credential and region chain on first use.
    async fn client(&self) -> &Client {
        self.client
            .get_or_init(|| async {
                let mut loader = aws_config::defaults(aws_config::BehaviorVersion::latest());
                if let Some(region) = &self.region {
                    loader = loader.region(aws_config::Region::new(region.clone()));
                }
                Client::new(&loader.load().await)
            })
            .await
    }

    /// The upsert statement for the exporter's engine and table.
    fn upsert_sql(&self) -> String {
        let table = &self.table;
        match self.engine {
            AuroraEngine::PostgreSql => format!(
                "INSERT INTO {table}
      (execution_arn, execution_name, function_name, status, start_time, end_time, duration_ms, record_json, emitted_at)
    VALUES
      (:execution_arn, :execution_name, :function_name, :status, :start_time::timestamptz, :end_time::timestamptz, :duration_ms, :record_json::jsonb, :emitted_at::timestamptz)
    ON CONFLICT (execution_arn) DO UPDATE SET
      status = EXCLUDED.status,
      end_time = EXCLUDED.end_time,
      duration_ms = EXCLUDED.duration_ms,
      record_json = EXCLUDED.record_json,
      emitted_at = EXCLUDED.emitted_at"
            ),
            AuroraEngine::MySql => format!(
                "INSERT INTO {table}
      (execution_arn, execution_name, function_name, status, start_time, end_time, duration_ms, record_json, emitted_at)
    VALUES
      (:execution_arn, :execution_name, :function_name, :status, :start_time, :end_time, :duration_ms, :record_json, :emitted_at)
    ON DUPLICATE KEY UPDATE
      status = VALUES(status),
      end_time = VALUES(end_time),
      duration_ms = VALUES(duration_ms),
      record_json = VALUES(record_json),
      emitted_at = VALUES(emitted_at)"
            ),
        }
    }

    /// The named parameters of the upsert statement for `record`.
    fn parameters(record: &TelemetryRecord) -> Result<Vec<SqlParameter>, BoxError> {
        // The scheduler truncated the record to the size of the array
        // rendering. So the same format is rendered here.
        let record_json = render_json(record, OperationsFormat::Array)?;
        Ok(vec![
            parameter("execution_arn", text(&record.execution_arn)),
            // An empty name is stored as NULL.
            parameter(
                "execution_name",
                optional_text(record.execution_name.as_deref()),
            ),
            parameter("function_name", text(&record.function_name)),
            parameter("status", text(status_text(record.status))),
            parameter("start_time", text(&record.start_time)),
            // An empty end time is stored as NULL.
            parameter("end_time", optional_text(record.end_time.as_deref())),
            parameter(
                "duration_ms",
                record
                    .duration_ms
                    .map_or(Field::IsNull(true), Field::LongValue),
            ),
            parameter("record_json", Field::StringValue(record_json)),
            parameter("emitted_at", text(&record.emitted_at)),
        ])
    }

    /// Upserts `record` with one `ExecuteStatement` request.
    async fn upsert(&self, record: &TelemetryRecord) -> Result<(), BoxError> {
        let parameters = Self::parameters(record)?;
        let client = self.client().await;
        client
            .execute_statement()
            .resource_arn(&self.resource_arn)
            .secret_arn(&self.secret_arn)
            .database(&self.database)
            .sql(self.upsert_sql())
            .set_parameters(Some(parameters))
            .send()
            .await
            .map_err(|error| call_error("ExecuteStatement", &error))?;
        Ok(())
    }
}

/// The status as the `status` column stores it.
fn status_text(status: ExecutionStatus) -> &'static str {
    match status {
        ExecutionStatus::Succeeded => "SUCCEEDED",
        ExecutionStatus::Failed => "FAILED",
        ExecutionStatus::Running => "RUNNING",
    }
}

fn text(value: &str) -> Field {
    Field::StringValue(value.to_owned())
}

/// A text field, or a null field for an absent or empty value.
fn optional_text(value: Option<&str>) -> Field {
    match value {
        Some(value) if !value.is_empty() => text(value),
        _ => Field::IsNull(true),
    }
}

fn parameter(name: &str, value: Field) -> SqlParameter {
    SqlParameter::builder().name(name).value(value).build()
}

/// The export error for a failed call of `operation`.
///
/// The `Display` text of an `SdkError` is a fixed string for each variant,
/// such as "service error". The service's error code and message are in
/// the error's sources. The scheduler logs only the `Display` text of an
/// export error. So the text here includes the sources.
fn call_error<E>(operation: &str, error: &SdkError<E>) -> BoxError
where
    E: std::error::Error + 'static,
{
    format!("{operation} failed: {}", DisplayErrorContext(error)).into()
}

/// Whether `name` is a valid SQL identifier for the table name: a letter or
/// underscore, then letters, digits and underscores, all ASCII.
fn is_valid_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

impl AuroraExporterBuilder {
    /// Sets the table name. The default is `durable_execution_telemetry`.
    ///
    /// The name goes into the SQL text, because a table name cannot be a
    /// statement parameter. So [`build`](Self::build) rejects a name that is
    /// not a letter or underscore followed by letters, digits and
    /// underscores.
    pub fn table(mut self, table: impl Into<String>) -> Self {
        self.table = table.into();
        self
    }

    /// Sets the region of the client the exporter builds. Without it, the
    /// default region chain applies. A [`client`](Self::client) ignores it.
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Sets the largest record, in bytes, before truncation. The default is
    /// 1,000,000.
    pub fn max_record_size_bytes(mut self, max: usize) -> Self {
        self.max_record_size_bytes = max;
        self
    }

    /// Uses `client` instead of building one from the default chain.
    ///
    /// The parameter type is re-exported at the crate root as
    /// [`RdsDataClient`](crate::RdsDataClient), so callers can name it
    /// without depending on `aws-sdk-rdsdata` directly.
    ///
    /// # Examples
    ///
    /// The SDK crate re-exports the configuration type. So this compiles
    /// with no direct `aws-config` or `aws-sdk-rdsdata` dependency:
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk::SdkConfig;
    /// use aws_durable_execution_sdk_telemetry_exporter::{
    ///     AuroraEngine, AuroraExporter, RdsDataClient,
    /// };
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    /// let config = SdkConfig::builder().build();
    /// let exporter = AuroraExporter::builder(
    ///     "arn:aws:rds:us-west-2:123456789012:cluster:telemetry",
    ///     "arn:aws:secretsmanager:us-west-2:123456789012:secret:telemetry",
    ///     "telemetry",
    ///     AuroraEngine::PostgreSql,
    /// )
    /// .client(RdsDataClient::new(&config))
    /// .build()?;
    /// # drop(exporter);
    /// # Ok(())
    /// # }
    /// ```
    pub fn client(mut self, client: crate::RdsDataClient) -> Self {
        self.client = Some(client);
        self
    }

    /// Builds the exporter.
    ///
    /// # Errors
    ///
    /// Returns an error when the table name is not a valid identifier. The
    /// error names the table.
    pub fn build(self) -> Result<AuroraExporter, BoxError> {
        if !is_valid_identifier(&self.table) {
            return Err(format!(
                "Invalid SQL identifier: \"{}\". Only letters, digits, and underscores are allowed, and the first character cannot be a digit.",
                self.table
            )
            .into());
        }
        Ok(AuroraExporter {
            resource_arn: self.resource_arn,
            secret_arn: self.secret_arn,
            database: self.database,
            table: self.table,
            engine: self.engine,
            region: self.region,
            max_record_size_bytes: self.max_record_size_bytes,
            client: tokio::sync::OnceCell::new_with(self.client),
        })
    }
}

impl TelemetryExporter for AuroraExporter {
    fn export<'a>(&'a self, record: &'a TelemetryRecord) -> ExportFuture<'a> {
        Box::pin(self.upsert(record))
    }

    fn max_record_size_bytes(&self) -> Option<usize> {
        Some(self.max_record_size_bytes)
    }

    fn operations_format(&self) -> OperationsFormat {
        OperationsFormat::Array
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::tests::{op, record};
    use aws_sdk_rdsdata::operation::execute_statement::{
        ExecuteStatementError, ExecuteStatementOutput,
    };
    use aws_sdk_rdsdata::types::error::BadRequestException;
    use aws_smithy_mocks::{RuleMode, mock, mock_client};
    use serde_json::Value;

    const CLUSTER: &str = "arn:aws:rds:us-west-2:123456789012:cluster:c";
    const SECRET: &str = "arn:aws:secretsmanager:us-west-2:123456789012:secret:s";

    fn builder(engine: AuroraEngine) -> AuroraExporterBuilder {
        AuroraExporter::builder(CLUSTER, SECRET, "db", engine)
    }

    #[expect(
        clippy::expect_used,
        reason = "a test builds an exporter with a valid table name"
    )]
    fn must_build(builder: AuroraExporterBuilder) -> AuroraExporter {
        builder.build().expect("the table name is valid")
    }

    fn exporter_for(engine: AuroraEngine) -> AuroraExporter {
        must_build(builder(engine))
    }

    fn field<'a>(parameters: &'a [SqlParameter], name: &str) -> Option<&'a Field> {
        parameters
            .iter()
            .find(|p| p.name() == Some(name))
            .and_then(SqlParameter::value)
    }

    fn error_text(result: Result<(), BoxError>) -> String {
        result
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default()
    }

    #[test]
    fn reports_its_limit_and_format() {
        let exporter = must_build(
            builder(AuroraEngine::MySql)
                .region("us-west-2")
                .max_record_size_bytes(10),
        );
        assert_eq!(exporter.max_record_size_bytes(), Some(10));
        assert_eq!(exporter.operations_format(), OperationsFormat::Array);
        assert_eq!(exporter.region.as_deref(), Some("us-west-2"));
        assert_eq!(
            exporter_for(AuroraEngine::MySql).max_record_size_bytes(),
            Some(1_000_000)
        );
    }

    #[test]
    fn identifiers_follow_the_sql_name_rule() {
        for valid in ["t", "_t", "T_1", "durable_execution_telemetry", "a1b2"] {
            assert!(is_valid_identifier(valid), "{valid}");
        }
        for invalid in ["", "1t", "t-1", "t.u", "t u", "t;drop", "tä", "t\n"] {
            assert!(!is_valid_identifier(invalid), "{invalid:?}");
        }
    }

    #[test]
    fn an_invalid_table_name_fails_the_build_and_names_the_table() {
        let error = builder(AuroraEngine::PostgreSql)
            .table("t; DROP TABLE x")
            .build()
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(error.contains("Invalid SQL identifier"), "{error}");
        assert!(error.contains("t; DROP TABLE x"), "{error}");
    }

    #[test]
    fn the_default_table_is_used_in_the_statement() {
        assert!(
            exporter_for(AuroraEngine::MySql)
                .upsert_sql()
                .starts_with("INSERT INTO durable_execution_telemetry\n")
        );
    }

    #[test]
    fn the_postgresql_statement_upserts_with_on_conflict() {
        let sql = must_build(builder(AuroraEngine::PostgreSql).table("runs")).upsert_sql();
        let expected = "INSERT INTO runs
      (execution_arn, execution_name, function_name, status, start_time, end_time, duration_ms, record_json, emitted_at)
    VALUES
      (:execution_arn, :execution_name, :function_name, :status, :start_time::timestamptz, :end_time::timestamptz, :duration_ms, :record_json::jsonb, :emitted_at::timestamptz)
    ON CONFLICT (execution_arn) DO UPDATE SET
      status = EXCLUDED.status,
      end_time = EXCLUDED.end_time,
      duration_ms = EXCLUDED.duration_ms,
      record_json = EXCLUDED.record_json,
      emitted_at = EXCLUDED.emitted_at";
        assert_eq!(sql, expected);
    }

    #[test]
    fn the_mysql_statement_upserts_with_on_duplicate_key() {
        let sql = must_build(builder(AuroraEngine::MySql).table("runs")).upsert_sql();
        let expected = "INSERT INTO runs
      (execution_arn, execution_name, function_name, status, start_time, end_time, duration_ms, record_json, emitted_at)
    VALUES
      (:execution_arn, :execution_name, :function_name, :status, :start_time, :end_time, :duration_ms, :record_json, :emitted_at)
    ON DUPLICATE KEY UPDATE
      status = VALUES(status),
      end_time = VALUES(end_time),
      duration_ms = VALUES(duration_ms),
      record_json = VALUES(record_json),
      emitted_at = VALUES(emitted_at)";
        assert_eq!(sql, expected);
    }

    #[test]
    fn status_text_matches_the_record_status() {
        for status in [
            ExecutionStatus::Running,
            ExecutionStatus::Succeeded,
            ExecutionStatus::Failed,
        ] {
            assert_eq!(
                serde_json::to_value(status).ok(),
                Some(Value::String(status_text(status).to_owned()))
            );
        }
    }

    #[tokio::test]
    async fn sends_the_statement_with_all_parameters() {
        let mut r = record(vec![op("a", "SUCCEEDED")]);
        r.status = ExecutionStatus::Succeeded;
        r.execution_name = Some("my-run".to_owned());
        r.end_time = Some("2024-01-01T00:00:05.000Z".to_owned());
        r.duration_ms = Some(5000);
        let expected_json = render_json(&r, OperationsFormat::Array).unwrap_or_default();
        let expected_arn = r.execution_arn.clone();
        let expected_function = r.function_name.clone();
        let expected_start = r.start_time.clone();
        let expected_emitted = r.emitted_at.clone();

        let execute = mock!(Client::execute_statement)
            .match_requests(move |req| {
                let p = req.parameters();
                let string = |name: &str| match field(p, name) {
                    Some(Field::StringValue(v)) => Some(v.clone()),
                    _ => None,
                };
                req.resource_arn() == Some(CLUSTER)
                    && req.secret_arn() == Some(SECRET)
                    && req.database() == Some("db")
                    && req.sql().is_some_and(|s| s.contains("ON CONFLICT"))
                    && p.len() == 9
                    && string("execution_arn") == Some(expected_arn.clone())
                    && string("execution_name").as_deref() == Some("my-run")
                    && string("function_name") == Some(expected_function.clone())
                    && string("status").as_deref() == Some("SUCCEEDED")
                    && string("start_time") == Some(expected_start.clone())
                    && string("end_time").as_deref() == Some("2024-01-01T00:00:05.000Z")
                    && matches!(field(p, "duration_ms"), Some(Field::LongValue(5000)))
                    && string("record_json") == Some(expected_json.clone())
                    && string("emitted_at") == Some(expected_emitted.clone())
            })
            .then_output(|| ExecuteStatementOutput::builder().build());
        let client = mock_client!(aws_sdk_rdsdata, RuleMode::MatchAny, [&execute]);
        let exporter = must_build(builder(AuroraEngine::PostgreSql).client(client));

        assert!(exporter.export(&r).await.is_ok());
        assert_eq!(execute.num_calls(), 1);
    }

    #[tokio::test]
    async fn absent_optional_values_are_sent_as_null() {
        let mut r = record(Vec::new());
        r.status = ExecutionStatus::Running;
        r.execution_name = None;
        r.end_time = None;
        r.duration_ms = None;
        let execute = mock!(Client::execute_statement)
            .match_requests(|req| {
                let p = req.parameters();
                req.sql().is_some_and(|s| s.contains("ON DUPLICATE KEY"))
                    && ["execution_name", "end_time", "duration_ms"]
                        .iter()
                        .all(|name| matches!(field(p, name), Some(Field::IsNull(true))))
                    && matches!(field(p, "status"), Some(Field::StringValue(s)) if s == "RUNNING")
            })
            .then_output(|| ExecuteStatementOutput::builder().build());
        let client = mock_client!(aws_sdk_rdsdata, RuleMode::MatchAny, [&execute]);
        let exporter = must_build(builder(AuroraEngine::MySql).client(client));

        assert!(exporter.export(&r).await.is_ok());
        assert_eq!(execute.num_calls(), 1);
    }

    #[tokio::test]
    async fn empty_name_and_end_time_are_sent_as_null() {
        let mut r = record(Vec::new());
        r.execution_name = Some(String::new());
        r.end_time = Some(String::new());
        r.duration_ms = Some(0);
        let execute = mock!(Client::execute_statement)
            .match_requests(|req| {
                let p = req.parameters();
                matches!(field(p, "execution_name"), Some(Field::IsNull(true)))
                    && matches!(field(p, "end_time"), Some(Field::IsNull(true)))
                    && matches!(field(p, "duration_ms"), Some(Field::LongValue(0)))
            })
            .then_output(|| ExecuteStatementOutput::builder().build());
        let client = mock_client!(aws_sdk_rdsdata, RuleMode::MatchAny, [&execute]);
        let exporter = must_build(builder(AuroraEngine::MySql).client(client));

        assert!(exporter.export(&r).await.is_ok());
        assert_eq!(execute.num_calls(), 1);
    }

    #[tokio::test]
    async fn record_json_holds_the_operations_array() {
        let r = record(vec![op("a", "SUCCEEDED")]);
        let execute = mock!(Client::execute_statement)
            .match_requests(|req| {
                let json = match field(req.parameters(), "record_json") {
                    Some(Field::StringValue(v)) => serde_json::from_str::<Value>(v).ok(),
                    _ => None,
                };
                json.is_some_and(|v| {
                    v.get("operations").is_some_and(Value::is_array)
                        && v.get("operationsByName").is_none()
                })
            })
            .then_output(|| ExecuteStatementOutput::builder().build());
        let client = mock_client!(aws_sdk_rdsdata, RuleMode::MatchAny, [&execute]);
        let exporter = must_build(builder(AuroraEngine::MySql).client(client));

        assert!(exporter.export(&r).await.is_ok());
        assert_eq!(execute.num_calls(), 1);
    }

    #[tokio::test]
    async fn a_service_error_fails_the_export_and_names_the_call() {
        let execute = mock!(Client::execute_statement).then_error(|| {
            ExecuteStatementError::BadRequestException(
                BadRequestException::builder()
                    .message("no such table")
                    .build(),
            )
        });
        let client = mock_client!(aws_sdk_rdsdata, RuleMode::MatchAny, [&execute]);
        let exporter = must_build(builder(AuroraEngine::MySql).client(client));

        let error = error_text(exporter.export(&record(Vec::new())).await);
        assert!(error.starts_with("ExecuteStatement failed: "), "{error}");
        assert!(error.contains("BadRequestException"), "{error}");
        assert!(error.contains("no such table"), "{error}");
        assert_eq!(execute.num_calls(), 1);
    }

    #[tokio::test]
    async fn flush_succeeds_without_a_call() {
        let exporter = exporter_for(AuroraEngine::MySql);
        assert!(exporter.flush().await.is_ok());
    }
}
