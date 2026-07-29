# AWS Durable Execution SDK for Rust

Rust SDK for AWS Lambda Durable Functions, enabling long-running
orchestrations that survive Lambda invocation timeouts through automatic
checkpointing and deterministic replay.

## Status

Pre-alpha and unpublished. The public API surface — steps, waits, callbacks,
child contexts, chained invoke, and map/parallel with deterministic replay —
is implemented, but the API is not yet frozen and may change. Quality gates
are active:

```sh
make check
```

## License

Apache-2.0. See [LICENSE](LICENSE).
