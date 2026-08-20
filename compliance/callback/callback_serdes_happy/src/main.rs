//! Conformance handler for requirement 4-15: callback with custom serdes (happy path).
//! A custom callback deserializer converts the ISO-8601 `timestamp` field of the
//! externally-delivered payload into Unix epoch seconds, so the value decodes
//! directly into the output type.

use aws_durable_execution_sdk_rust as durable;
use durable::{BoxError, Serdes};
use serde::{Deserialize, Serialize};

/// Raw callback payload from the external system (timestamp as ISO-8601 string).
#[derive(Deserialize)]
struct CallbackPayload {
    id: String,
    message: String,
    timestamp: String,
}

/// Output returned by the handler.
#[derive(Serialize)]
struct ResultOutput {
    received: ReceivedData,
}

/// Deserialized callback data with timestamp as epoch seconds.
#[derive(Serialize, Deserialize)]
struct ReceivedData {
    id: String,
    message: String,
    timestamp: i64,
}

/// Callback deserializer: rewrites the raw payload so the ISO-8601 `timestamp`
/// becomes epoch seconds, letting the payload decode straight into
/// [`ReceivedData`]. The callback payload is produced externally, so only the
/// deserialize side is meaningful.
#[derive(Debug)]
struct TimestampSerdes;

impl Serdes<ReceivedData> for TimestampSerdes {
    async fn serialize(
        &self,
        value: ReceivedData,
        _context: durable::serdes::SerdesContext,
    ) -> Result<String, BoxError> {
        // The callback payload is produced externally; the SDK never
        // serializes a value on the way out. Plain JSON keeps the impl total.
        Ok(serde_json::to_string(&value)?)
    }

    async fn deserialize(
        &self,
        wire: String,
        _context: durable::serdes::SerdesContext,
    ) -> Result<ReceivedData, BoxError> {
        let raw: CallbackPayload = serde_json::from_str(&wire)?;
        let timestamp = parse_iso_timestamp(&raw.timestamp)?;
        Ok(ReceivedData {
            id: raw.id,
            message: raw.message,
            timestamp,
        })
    }
}

/// Parse an ISO-8601 timestamp into Unix epoch seconds (simple implementation
/// matching the conformance expectation: 2026-01-01T00:00:00.000Z → 1767225600).
fn parse_iso_timestamp(s: &str) -> Result<i64, Box<dyn std::error::Error + Send + Sync>> {
    // Expected format: YYYY-MM-DDTHH:MM:SS.sssZ
    // We'll parse it manually since we can't add chrono as a dependency.
    let s = s.trim_end_matches('Z');
    let parts: Vec<&str> = s.split('T').collect();
    let date_parts: Vec<i64> = parts
        .first()
        .ok_or("missing date")?
        .split('-')
        .map(|p| p.parse::<i64>())
        .collect::<Result<Vec<_>, _>>()?;
    let time_parts: Vec<&str> = parts.get(1).ok_or("missing time")?.split(':').collect();

    let year = *date_parts.first().ok_or("missing year")?;
    let month = *date_parts.get(1).ok_or("missing month")?;
    let day = *date_parts.get(2).ok_or("missing day")?;
    let hour: i64 = time_parts.first().ok_or("missing hour")?.parse()?;
    let minute: i64 = time_parts.get(1).ok_or("missing minute")?.parse()?;
    let sec_str = *time_parts.get(2).ok_or("missing second")?;
    let second: i64 = sec_str.split('.').next().ok_or("bad seconds")?.parse()?;

    // Days from year 1970 to `year` (simplified, no leap-second accuracy needed)
    let mut days: i64 = 0;
    for y in 1970..year {
        days += if is_leap(y) { 366 } else { 365 };
    }
    let month_days = [
        31,
        28 + i64::from(is_leap(year)),
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    for m in 0..(month - 1) as usize {
        days += month_days.get(m).copied().unwrap_or(30);
    }
    days += day - 1;

    Ok(days * 86400 + hour * 3600 + minute * 60 + second)
}

const fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        // The callback serdes converts the raw payload's ISO timestamp into
        // epoch seconds, so the payload decodes directly into ReceivedData.
        let cb = ctx
            .create_callback::<ReceivedData>()
            .name(&name)
            .serdes(TimestampSerdes)
            .await?;
        let received = cb.result().await?;
        let output = ResultOutput { received };
        Ok(output)
    })
    .await
}
