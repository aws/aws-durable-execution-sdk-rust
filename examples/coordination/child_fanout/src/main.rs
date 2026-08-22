//! Child-context fan-out: spawned children joined with `try_join_all`.
//!
//! `map` fans one closure out over a homogeneous collection, and `parallel`
//! runs a fixed set of named branches. When the branches are heterogeneous
//! AND their number is decided at run time — here, the line items of an
//! incoming order, each fulfilled by a different multi-step workflow — fan
//! out explicitly instead: spawn one child context per unit of work with
//! `.spawn()`, collect the running
//! [`DurableFuture`](aws_durable_execution_sdk_rust::DurableFuture)s, and
//! join them with [`try_join_all`]. Each child performs multi-step work of
//! its own, exactly as in the `child_basic` example, and all children make
//! progress concurrently.
//!
//! The determinism rule this demonstrates: children are created in
//! deterministic source order, and each child's operation ID is minted
//! synchronously when its builder is created, before any child runs. This
//! example makes that ordering explicit with two passes: it first collects
//! every child builder into a `Vec`, fixing all the operation IDs, and only
//! then spawns them. Each child's operations then live in the child's own
//! namespace. Concurrent execution therefore cannot reorder operation
//! identities, no matter which child finishes first. What you must supply
//! is the deterministic order of creation: fan out over a `Vec`, never a
//! `HashMap`, whose iteration order changes between runs and would replay a
//! different history.
//!
//! This example fulfills an order. The event carries the order's line
//! items, so the input decides at run time how many children to spawn, and
//! each item kind runs a different workflow: a physical item reserves stock
//! and schedules a shipment, a digital item issues a license and delivers
//! the download, and a subscription activates the plan, provisions its
//! entitlements, and schedules the first renewal. The parent joins the
//! confirmations and returns one fulfillment summary.
//!
//! [`try_join_all`]: aws_durable_execution_sdk_rust::DurableContext::try_join_all

use aws_durable_execution_sdk_rust as durable;
use serde::Deserialize;

/// Handler input: an order to fulfill.
#[derive(Debug, Deserialize)]
struct Order {
    /// Identifier echoed in the fulfillment summary.
    order_id: String,
    /// Line items, one spawned child per item. Arriving as a `Vec`, the
    /// items carry the deterministic order the fan-out below relies on.
    items: Vec<LineItem>,
}

/// One unit of fulfillment work. Each variant runs a different multi-step
/// workflow in its own child context.
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum LineItem {
    /// A stocked good: reserve inventory, then schedule a shipment.
    Physical {
        /// Stock-keeping unit to reserve and ship.
        sku: String,
        /// Units to reserve.
        quantity: u32,
    },
    /// A downloadable good: issue a license, then deliver the link.
    Digital {
        /// Stock-keeping unit to license.
        sku: String,
    },
    /// A recurring plan: activate it, provision entitlements, schedule the
    /// first renewal.
    Subscription {
        /// Plan to activate.
        plan: String,
        /// Term length in months.
        months: u32,
    },
}

impl LineItem {
    /// Short slug naming the workflow in the child operation's name.
    fn label(&self) -> &'static str {
        match self {
            LineItem::Physical { .. } => "physical",
            LineItem::Digital { .. } => "digital",
            LineItem::Subscription { .. } => "subscription",
        }
    }
}

/// Fulfills a physical item: reserve stock, then schedule a shipment.
async fn fulfill_physical(
    child: durable::DurableContext,
    sku: String,
    quantity: u32,
) -> Result<String, durable::BoxError> {
    let reservation = child
        .step({
            let sku = sku.clone();
            // Deterministic stand-in for a real inventory call.
            move |_| async move { Ok(format!("rsv-{sku}-x{quantity}")) }
        })
        .name("reserve-stock")
        .await?;
    let shipment = child
        .step(move |_| async move { Ok(format!("shp-{reservation}")) })
        .name("schedule-shipment")
        .await?;
    Ok(format!("{sku}: shipping as {shipment}"))
}

/// Fulfills a digital item: issue a license, then deliver the download.
async fn fulfill_digital(
    child: durable::DurableContext,
    sku: String,
) -> Result<String, durable::BoxError> {
    let license = child
        .step({
            let sku = sku.clone();
            // Deterministic stand-in for a real licensing call.
            move |_| async move { Ok(format!("lic-{sku}")) }
        })
        .name("issue-license")
        .await?;
    let delivery = child
        .step({
            let license = license.clone();
            move |_| async move { Ok(format!("downloads.example.com/{license}")) }
        })
        .name("deliver-download")
        .await?;
    Ok(format!("{sku}: license {license} at {delivery}"))
}

/// Fulfills a subscription: activate the plan, provision its entitlements,
/// then schedule the first renewal.
async fn fulfill_subscription(
    child: durable::DurableContext,
    plan: String,
    months: u32,
) -> Result<String, durable::BoxError> {
    let account = child
        .step({
            let plan = plan.clone();
            // Deterministic stand-in for a real billing call.
            move |_| async move { Ok(format!("acct-{plan}")) }
        })
        .name("activate-plan")
        .await?;
    let entitlements = child
        .step({
            let account = account.clone();
            move |_| async move { Ok(format!("ent-{account}")) }
        })
        .name("provision-entitlements")
        .await?;
    let renewal = child
        .step(move |_| async move { Ok(format!("renewal-{account}-month-{months}")) })
        .name("schedule-renewal")
        .await?;
    Ok(format!("{plan}: {entitlements}, first {renewal}"))
}

/// Fulfills each line item in its own spawned child context and returns a
/// summary of the whole order.
async fn handler(event: Order, ctx: durable::DurableContext) -> Result<String, durable::BoxError> {
    let Order { order_id, items } = event;

    // The event's Vec fixes both the child count and the fan-out order:
    // every execution of this order creates the same children in the same
    // source order, so each child claims the same operation ID on every
    // run. Re-keying the items through a HashMap here would break replay.
    let builders: Vec<_> = items
        .into_iter()
        .enumerate()
        .map(|(index, item)| {
            let label = item.label();
            // First pass: create one builder per item. `run_in_child_context`
            // mints the child's operation ID synchronously right here, so
            // after this pass every child's identity is fixed — and no child
            // has run yet.
            ctx.run_in_child_context(move |child| async move {
                match item {
                    LineItem::Physical { sku, quantity } => {
                        fulfill_physical(child, sku, quantity).await
                    }
                    LineItem::Digital { sku } => fulfill_digital(child, sku).await,
                    LineItem::Subscription { plan, months } => {
                        fulfill_subscription(child, plan, months).await
                    }
                }
            })
            .name(format!("fulfill-{index}-{label}"))
        })
        .collect();

    // Second pass: start them. All IDs were minted above, and each child's
    // own steps are numbered in the child's namespace, so the children can
    // run and finish in any order without colliding.
    let children: Vec<durable::DurableFuture<String>> = builders
        .into_iter()
        .map(durable::builders::ChildBuilder::spawn)
        .collect();

    // Gather every confirmation, failing fast if any item fails. The
    // results come back in input order, matching the event's item order.
    let confirmations = ctx
        .try_join_all(children)
        .name("gather-fulfillments")
        .await?;
    Ok(format!("{order_id}: {}", confirmations.join("; ")))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
