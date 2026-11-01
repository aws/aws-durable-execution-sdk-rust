//! The public Lambda transport is an `ExecutionClient` a caller can build
//! and pass back to the SDK.

use std::sync::Arc;

use aws_durable_execution_sdk::{
    ExecutionClient, LambdaClient, LambdaExecutionClient, Options, SdkConfig,
};

#[test]
fn public_lambda_transport_is_an_execution_client() {
    let client = LambdaClient::new(&SdkConfig::builder().build());
    let transport = LambdaExecutionClient::new(client);
    assert_eq!(format!("{transport:?}"), "LambdaExecutionClient");

    let transport: Arc<dyn ExecutionClient> = Arc::new(transport);
    let options = Options::builder().execution_client(transport).build();
    assert!(options.is_ok(), "{options:?}");
}
