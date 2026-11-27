//! Writes records to an Amazon Redshift table with the Redshift Data API.

use aws_durable_execution_sdk::BoxError;
use aws_sdk_redshiftdata::Client;
use aws_sdk_redshiftdata::error::DisplayErrorContext;
use aws_sdk_redshiftdata::types::SqlParameter;

use crate::exporter::{ExportFuture, TelemetryExporter};
use crate::record::{ExecutionStatus, TelemetryRecord};
use crate::render::{OperationsFormat, render_json};

/// The default record size limit, in bytes. It differs from the crate-wide
/// `DEFAULT_MAX_RECORD_SIZE_BYTES`.
const DEFAULT_REDSHIFT_MAX_RECORD_SIZE_BYTES: usize = 1_000_000;

/// The default table name.
const DEFAULT_TABLE: &str = "durable_execution_telemetry";

/// The default schema name.
const DEFAULT_SCHEMA: &str = "public";

/// Writes each record as one row of a Redshift table.
///
/// The exporter runs one `MERGE` statement per record with the Redshift
/// Data API. The statement matches rows on `execution_arn`. A record of an
/// execution that has a row updates the row's `status`, `end_time`,
/// `duration_ms`, `record_json` and `emitted_at`. A record of any other
/// execution inserts a row. So a table holds one row per execution.
///
/// The table must exist, with these columns:
///
/// | Column | Type |
/// | --- | --- |
/// | `execution_arn` | `VARCHAR` |
/// | `execution_name` | `VARCHAR` |
/// | `function_name` | `VARCHAR` |
/// | `status` | `VARCHAR` |
/// | `start_time` | `TIMESTAMPTZ` |
/// | `end_time` | `TIMESTAMPTZ` |
/// | `duration_ms` | `BIGINT` |
/// | `record_json` | `SUPER` |
/// | `emitted_at` | `TIMESTAMPTZ` |
///
/// The Data API runs a statement asynchronously. The exporter does not wait
/// for the statement to finish. So an export succeeds when Redshift accepts
/// the statement, and a statement that later fails is not reported.
///
/// The function's role needs `redshift-data:ExecuteStatement`. It also needs
/// `redshift-serverless:GetCredentials` for Redshift Serverless, or
/// `redshift:GetClusterCredentialsWithIAM` for a provisioned cluster.
#[derive(Debug)]
pub struct RedshiftExporter {
    database: String,
    workgroup_name: Option<String>,
    cluster_identifier: Option<String>,
    db_user: Option<String>,
    secret_arn: Option<String>,
    qualified_table: String,
    region: Option<String>,
    max_record_size_bytes: usize,
    client: tokio::sync::OnceCell<Client>,
}

/// Builds a [`RedshiftExporter`].
#[derive(Debug)]
#[must_use]
pub struct RedshiftExporterBuilder {
    database: String,
    workgroup_name: Option<String>,
    cluster_identifier: Option<String>,
    db_user: Option<String>,
    secret_arn: Option<String>,
    table: String,
    schema: String,
    region: Option<String>,
    max_record_size_bytes: usize,
    client: Option<Client>,
}

/// The SQL of one export and its bound parameters.
#[derive(Debug, PartialEq, Eq)]
struct Statement {
    sql: String,
    /// Each parameter's name and value, in binding order.
    parameters: Vec<(&'static str, String)>,
}

impl RedshiftExporter {
    /// A builder for an exporter that writes to the database `database`.
    ///
    /// The builder needs a [`workgroup_name`](RedshiftExporterBuilder::workgroup_name)
    /// (Redshift Serverless) or a
    /// [`cluster_identifier`](RedshiftExporterBuilder::cluster_identifier)
    /// (provisioned cluster) before it can build.
    pub fn builder(database: impl Into<String>) -> RedshiftExporterBuilder {
        RedshiftExporterBuilder {
            database: database.into(),
            workgroup_name: None,
            cluster_identifier: None,
            db_user: None,
            secret_arn: None,
            table: DEFAULT_TABLE.to_owned(),
            schema: DEFAULT_SCHEMA.to_owned(),
            region: None,
            max_record_size_bytes: DEFAULT_REDSHIFT_MAX_RECORD_SIZE_BYTES,
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

    /// The statement that upserts `record`.
    ///
    /// The Data API accepts neither a null nor an empty string as a
    /// parameter value. So an absent `execution_name`, `end_time` or
    /// `duration_ms` is a typed `NULL` literal in the statement, and no
    /// parameter is bound for it.
    ///
    /// The `MERGE` joins on a column of the source subquery. Redshift
    /// rejects a join on a parameter or a constant, because it plans that
    /// join as a nested loop.
    fn statement(&self, record: &TelemetryRecord) -> Result<Statement, BoxError> {
        let table = &self.qualified_table;
        let mut parameters = vec![
            ("execution_arn", record.execution_arn.clone()),
            ("function_name", record.function_name.clone()),
            ("status", status_text(record.status).to_owned()),
            ("start_time", record.start_time.clone()),
            (
                "record_json",
                render_json(record, self.operations_format())?,
            ),
            ("emitted_at", record.emitted_at.clone()),
        ];

        let execution_name = match record.execution_name.as_deref() {
            Some(name) if !name.is_empty() => {
                parameters.push(("execution_name", name.to_owned()));
                ":execution_name::varchar"
            }
            _ => "NULL::varchar",
        };
        let end_time = match record.end_time.as_deref() {
            Some(time) if !time.is_empty() => {
                parameters.push(("end_time", time.to_owned()));
                ":end_time::timestamptz"
            }
            _ => "NULL::timestamptz",
        };
        let duration_ms = if let Some(duration) = record.duration_ms {
            parameters.push(("duration_ms", duration.to_string()));
            ":duration_ms::bigint"
        } else {
            "NULL::bigint"
        };

        let sql = format!(
            "MERGE INTO {table} USING (
      SELECT
        :execution_arn::varchar AS execution_arn,
        {execution_name} AS execution_name,
        :function_name::varchar AS function_name,
        :status::varchar AS status,
        :start_time::timestamptz AS start_time,
        {end_time} AS end_time,
        {duration_ms} AS duration_ms,
        JSON_PARSE(:record_json) AS record_json,
        :emitted_at::timestamptz AS emitted_at
    ) AS src
    ON {table}.execution_arn = src.execution_arn
    WHEN MATCHED THEN UPDATE SET
      status = src.status,
      end_time = src.end_time,
      duration_ms = src.duration_ms,
      record_json = src.record_json,
      emitted_at = src.emitted_at
    WHEN NOT MATCHED THEN INSERT
      (execution_arn, execution_name, function_name, status, start_time, end_time, duration_ms, record_json, emitted_at)
    VALUES
      (src.execution_arn, src.execution_name, src.function_name, src.status, src.start_time, src.end_time, src.duration_ms, src.record_json, src.emitted_at)"
        );
        Ok(Statement { sql, parameters })
    }

    /// Starts the statement that upserts `record`. The call returns when
    /// Redshift accepts the statement, not when it finishes.
    async fn put(&self, record: &TelemetryRecord) -> Result<(), BoxError> {
        let Statement { sql, parameters } = self.statement(record)?;
        let parameters = parameters.into_iter().map(|(name, value)| {
            #[expect(
                clippy::expect_used,
                reason = "build fails only when the name or value is missing, and both are set"
            )]
            SqlParameter::builder()
                .name(name)
                .value(value)
                .build()
                .expect("name and value are set")
        });
        self.client()
            .await
            .execute_statement()
            .set_workgroup_name(self.workgroup_name.clone())
            .set_cluster_identifier(self.cluster_identifier.clone())
            .database(&self.database)
            .set_db_user(self.db_user.clone())
            .set_secret_arn(self.secret_arn.clone())
            .sql(sql)
            .set_parameters(Some(parameters.collect()))
            .send()
            .await
            // The `Display` text of an `SdkError` is a fixed string for each
            // variant. The service's error code and message are in its
            // sources, and the scheduler logs only the `Display` text. So the
            // text here includes the sources.
            .map_err(|error| format!("ExecuteStatement failed: {}", DisplayErrorContext(&error)))?;
        Ok(())
    }
}

/// The `status` column text of an execution status.
fn status_text(status: ExecutionStatus) -> &'static str {
    match status {
        ExecutionStatus::Running => "RUNNING",
        ExecutionStatus::Succeeded => "SUCCEEDED",
        ExecutionStatus::Failed => "FAILED",
    }
}

/// Whether `name` is a SQL identifier this exporter accepts: a letter or
/// underscore, then letters, digits or underscores.
fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Checks that `name` is an identifier. `what` names the option in the
/// error.
fn check_identifier(what: &str, name: &str) -> Result<(), BoxError> {
    if is_identifier(name) {
        Ok(())
    } else {
        Err(format!(
            "invalid {what} {name:?}: use only letters, digits and underscores, and do not start with a digit"
        )
        .into())
    }
}

impl RedshiftExporterBuilder {
    /// Sets the Redshift Serverless workgroup name.
    pub fn workgroup_name(mut self, name: impl Into<String>) -> Self {
        self.workgroup_name = Some(name.into());
        self
    }

    /// Sets the identifier of a provisioned cluster.
    pub fn cluster_identifier(mut self, identifier: impl Into<String>) -> Self {
        self.cluster_identifier = Some(identifier.into());
        self
    }

    /// Sets the database user. A provisioned cluster uses it to get
    /// temporary credentials. Redshift Serverless does not need it.
    pub fn db_user(mut self, user: impl Into<String>) -> Self {
        self.db_user = Some(user.into());
        self
    }

    /// Sets the ARN of the Secrets Manager secret that holds the
    /// credentials. It is an alternative to
    /// [`db_user`](Self::db_user) for a provisioned cluster.
    pub fn secret_arn(mut self, arn: impl Into<String>) -> Self {
        self.secret_arn = Some(arn.into());
        self
    }

    /// Sets the table name. The default is `durable_execution_telemetry`.
    /// [`build`](Self::build) fails unless the name matches
    /// `^[a-zA-Z_][a-zA-Z0-9_]*$`.
    pub fn table(mut self, table: impl Into<String>) -> Self {
        self.table = table.into();
        self
    }

    /// Sets the schema name. The default is `public`.
    /// [`build`](Self::build) fails unless the name matches
    /// `^[a-zA-Z_][a-zA-Z0-9_]*$`.
    pub fn schema(mut self, schema: impl Into<String>) -> Self {
        self.schema = schema.into();
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
    /// [`RedshiftDataClient`](crate::RedshiftDataClient), so callers can
    /// name it without depending on `aws-sdk-redshiftdata` directly.
    ///
    /// # Examples
    ///
    /// The SDK crate re-exports the configuration type. So this compiles
    /// with no direct `aws-config` or `aws-sdk-redshiftdata` dependency:
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk::SdkConfig;
    /// use aws_durable_execution_sdk_telemetry_exporter::{
    ///     RedshiftDataClient, RedshiftExporter,
    /// };
    ///
    /// let config = SdkConfig::builder().build();
    /// let exporter = RedshiftExporter::builder("dev")
    ///     .workgroup_name("my-workgroup")
    ///     .client(RedshiftDataClient::new(&config))
    ///     .build();
    /// # drop(exporter);
    /// ```
    pub fn client(mut self, client: crate::RedshiftDataClient) -> Self {
        self.client = Some(client);
        self
    }

    /// Builds the exporter.
    ///
    /// # Errors
    ///
    /// Returns an error when neither a workgroup name nor a cluster
    /// identifier is set, or when the table or schema name is not a SQL
    /// identifier.
    pub fn build(self) -> Result<RedshiftExporter, BoxError> {
        let named = |value: &Option<String>| value.as_deref().is_some_and(|v| !v.is_empty());
        if !named(&self.workgroup_name) && !named(&self.cluster_identifier) {
            return Err("RedshiftExporter needs a workgroup_name or a cluster_identifier".into());
        }
        check_identifier("table", &self.table)?;
        check_identifier("schema", &self.schema)?;
        Ok(RedshiftExporter {
            database: self.database,
            workgroup_name: self.workgroup_name,
            cluster_identifier: self.cluster_identifier,
            db_user: self.db_user,
            secret_arn: self.secret_arn,
            qualified_table: format!("{}.{}", self.schema, self.table),
            region: self.region,
            max_record_size_bytes: self.max_record_size_bytes,
            client: tokio::sync::OnceCell::new_with(self.client),
        })
    }
}

impl TelemetryExporter for RedshiftExporter {
    fn export<'a>(&'a self, record: &'a TelemetryRecord) -> ExportFuture<'a> {
        Box::pin(self.put(record))
    }

    fn max_record_size_bytes(&self) -> Option<usize> {
        Some(self.max_record_size_bytes)
    }

    fn operations_format(&self) -> OperationsFormat {
        OperationsFormat::Array
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "a test builds its fixtures with options that are valid, and a failure fails the test"
)]
mod tests {
    use super::*;
    use crate::render::tests::{op, record};
    use aws_sdk_redshiftdata::operation::execute_statement::{
        ExecuteStatementError, ExecuteStatementOutput,
    };
    use aws_sdk_redshiftdata::types::error::ValidationException;
    use aws_smithy_mocks::{RuleMode, mock, mock_client};
    use std::collections::HashMap;

    fn exporter() -> RedshiftExporter {
        RedshiftExporter::builder("dev")
            .workgroup_name("wg")
            .build()
            .expect("valid options")
    }

    fn param<'a>(statement: &'a Statement, name: &str) -> Option<&'a str> {
        statement
            .parameters
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| v.as_str())
    }

    fn error_text<T>(result: Result<T, BoxError>) -> String {
        result.err().map(|e| e.to_string()).unwrap_or_default()
    }

    #[test]
    fn reports_its_limit_and_format() {
        let custom = RedshiftExporter::builder("dev")
            .cluster_identifier("c")
            .region("us-west-2")
            .max_record_size_bytes(10)
            .build()
            .expect("valid options");
        assert_eq!(custom.max_record_size_bytes(), Some(10));
        assert_eq!(custom.operations_format(), OperationsFormat::Array);
        assert_eq!(custom.region.as_deref(), Some("us-west-2"));
        assert_eq!(exporter().max_record_size_bytes(), Some(1_000_000));
    }

    #[test]
    fn build_needs_a_workgroup_or_a_cluster() {
        let error = error_text(RedshiftExporter::builder("dev").build());
        assert!(error.contains("workgroup_name"), "{error}");
        assert!(error.contains("cluster_identifier"), "{error}");
        let empty = RedshiftExporter::builder("dev")
            .workgroup_name("")
            .cluster_identifier("")
            .build();
        assert!(empty.is_err());
        assert!(
            RedshiftExporter::builder("dev")
                .cluster_identifier("c")
                .build()
                .is_ok()
        );
    }

    #[test]
    fn table_and_schema_are_qualified_with_defaults() {
        assert_eq!(
            exporter().qualified_table,
            "public.durable_execution_telemetry"
        );
        let custom = RedshiftExporter::builder("dev")
            .workgroup_name("wg")
            .schema("_s1")
            .table("T_2")
            .build()
            .expect("valid options");
        assert_eq!(custom.qualified_table, "_s1.T_2");
    }

    #[test]
    fn table_and_schema_must_be_identifiers() {
        for bad in ["", "1a", "a-b", "a b", "a;drop table x", "a.b", "é", "a\n"] {
            let table = RedshiftExporter::builder("dev")
                .workgroup_name("wg")
                .table(bad)
                .build();
            assert!(
                error_text(table).contains("invalid table"),
                "table {bad:?} was accepted"
            );
            let schema = RedshiftExporter::builder("dev")
                .workgroup_name("wg")
                .schema(bad)
                .build();
            assert!(
                error_text(schema).contains("invalid schema"),
                "schema {bad:?} was accepted"
            );
        }
    }

    #[test]
    fn a_full_record_binds_every_parameter() {
        let r = record(vec![op("a", "SUCCEEDED")]);
        let statement = exporter().statement(&r).expect("renders");
        let names: Vec<&str> = statement.parameters.iter().map(|(n, _)| *n).collect();
        assert_eq!(
            names,
            [
                "execution_arn",
                "function_name",
                "status",
                "start_time",
                "record_json",
                "emitted_at",
                "execution_name",
                "end_time",
                "duration_ms",
            ]
        );
        assert_eq!(param(&statement, "execution_arn"), Some("arn"));
        assert_eq!(param(&statement, "function_name"), Some("fn"));
        assert_eq!(param(&statement, "status"), Some("SUCCEEDED"));
        assert_eq!(
            param(&statement, "start_time"),
            Some("2026-01-01T00:00:00.000Z")
        );
        assert_eq!(
            param(&statement, "emitted_at"),
            Some("2026-01-01T00:00:01.000Z")
        );
        assert_eq!(param(&statement, "execution_name"), Some("exec"));
        assert_eq!(
            param(&statement, "end_time"),
            Some("2026-01-01T00:00:01.000Z")
        );
        assert_eq!(param(&statement, "duration_ms"), Some("1000"));
        assert!(statement.sql.contains(":execution_name::varchar AS"));
        assert!(statement.sql.contains(":end_time::timestamptz AS"));
        assert!(statement.sql.contains(":duration_ms::bigint AS"));
        assert!(!statement.sql.contains("NULL::"));
    }

    #[test]
    fn record_json_is_the_array_form() {
        let r = record(vec![op("a", "SUCCEEDED")]);
        let statement = exporter().statement(&r).expect("renders");
        let json: serde_json::Value =
            serde_json::from_str(param(&statement, "record_json").expect("bound"))
                .expect("valid JSON");
        assert_eq!(json.pointer("/operations/0/name"), Some(&"a".into()));
        assert!(json.get("operationsByName").is_none());
        assert_eq!(json.pointer("/executionArn"), Some(&"arn".into()));
    }

    #[test]
    fn absent_values_are_typed_null_literals_without_parameters() {
        let mut r = record(Vec::new());
        r.execution_name = None;
        r.end_time = None;
        r.duration_ms = None;
        let statement = exporter().statement(&r).expect("renders");
        for name in ["execution_name", "end_time", "duration_ms"] {
            assert_eq!(param(&statement, name), None, "{name}");
        }
        assert!(statement.sql.contains("NULL::varchar AS execution_name"));
        assert!(statement.sql.contains("NULL::timestamptz AS end_time"));
        assert!(statement.sql.contains("NULL::bigint AS duration_ms"));
        assert!(!statement.sql.contains(":execution_name"));
        assert!(!statement.sql.contains(":end_time"));
        assert!(!statement.sql.contains(":duration_ms"));
    }

    #[test]
    fn an_empty_name_or_end_time_is_null_and_a_zero_duration_is_bound() {
        let mut r = record(Vec::new());
        r.execution_name = Some(String::new());
        r.end_time = Some(String::new());
        r.duration_ms = Some(0);
        let statement = exporter().statement(&r).expect("renders");
        assert_eq!(param(&statement, "execution_name"), None);
        assert_eq!(param(&statement, "end_time"), None);
        assert_eq!(param(&statement, "duration_ms"), Some("0"));
        assert!(statement.sql.contains("NULL::varchar AS execution_name"));
        assert!(statement.sql.contains("NULL::timestamptz AS end_time"));
    }

    #[test]
    fn the_status_column_holds_the_record_status() {
        for (status, text) in [
            (ExecutionStatus::Running, "RUNNING"),
            (ExecutionStatus::Succeeded, "SUCCEEDED"),
            (ExecutionStatus::Failed, "FAILED"),
        ] {
            let mut r = record(Vec::new());
            r.status = status;
            let statement = exporter().statement(&r).expect("renders");
            assert_eq!(param(&statement, "status"), Some(text));
        }
    }

    #[test]
    fn the_merge_joins_on_the_source_column_in_the_qualified_table() {
        let custom = RedshiftExporter::builder("dev")
            .workgroup_name("wg")
            .schema("s")
            .table("t")
            .build()
            .expect("valid options");
        let sql = custom.statement(&record(Vec::new())).expect("renders").sql;
        assert!(sql.starts_with("MERGE INTO s.t USING ("), "{sql}");
        assert!(sql.contains("ON s.t.execution_arn = src.execution_arn"));
        assert!(sql.contains("JSON_PARSE(:record_json) AS record_json"));
        assert!(sql.contains("WHEN MATCHED THEN UPDATE SET"));
        assert!(sql.contains("WHEN NOT MATCHED THEN INSERT"));
    }

    #[tokio::test]
    async fn export_sends_one_execute_statement_with_the_options() {
        let execute = mock!(Client::execute_statement)
            .match_requests(|req| {
                let parameters: HashMap<&str, &str> = req
                    .parameters()
                    .iter()
                    .map(|p| (p.name(), p.value()))
                    .collect();
                req.workgroup_name() == Some("wg")
                    && req.cluster_identifier() == Some("cl")
                    && req.database() == Some("dev")
                    && req.db_user() == Some("user")
                    && req.secret_arn() == Some("secret")
                    && req
                        .sql()
                        .is_some_and(|sql| sql.starts_with("MERGE INTO s.t "))
                    && req.parameters().len() == 9
                    && parameters.get("execution_arn") == Some(&"arn")
                    && parameters.get("duration_ms") == Some(&"1000")
            })
            .then_output(|| ExecuteStatementOutput::builder().build());
        let client = mock_client!(aws_sdk_redshiftdata, RuleMode::MatchAny, [&execute]);
        let exporter = RedshiftExporter::builder("dev")
            .workgroup_name("wg")
            .cluster_identifier("cl")
            .db_user("user")
            .secret_arn("secret")
            .schema("s")
            .table("t")
            .client(client)
            .build()
            .expect("valid options");

        assert!(exporter.export(&record(Vec::new())).await.is_ok());
        assert_eq!(execute.num_calls(), 1);
    }

    #[tokio::test]
    async fn unset_options_are_left_out_of_the_request() {
        let execute = mock!(Client::execute_statement)
            .match_requests(|req| {
                req.workgroup_name() == Some("wg")
                    && req.cluster_identifier().is_none()
                    && req.db_user().is_none()
                    && req.secret_arn().is_none()
            })
            .then_output(|| ExecuteStatementOutput::builder().build());
        let client = mock_client!(aws_sdk_redshiftdata, RuleMode::MatchAny, [&execute]);
        let exporter = RedshiftExporter::builder("dev")
            .workgroup_name("wg")
            .client(client)
            .build()
            .expect("valid options");

        assert!(exporter.export(&record(Vec::new())).await.is_ok());
        assert_eq!(execute.num_calls(), 1);
    }

    #[tokio::test]
    async fn a_failed_call_names_the_call_and_the_service_error() {
        let execute = mock!(Client::execute_statement).then_error(|| {
            ExecuteStatementError::ValidationException(
                ValidationException::builder()
                    .message("bad workgroup")
                    .build(),
            )
        });
        let client = mock_client!(aws_sdk_redshiftdata, RuleMode::MatchAny, [&execute]);
        let exporter = RedshiftExporter::builder("dev")
            .workgroup_name("wg")
            .client(client)
            .build()
            .expect("valid options");

        let error = error_text(exporter.export(&record(Vec::new())).await);
        assert!(error.starts_with("ExecuteStatement failed: "), "{error}");
        assert!(error.contains("ValidationException"), "{error}");
        assert!(error.contains("bad workgroup"), "{error}");
    }
}
