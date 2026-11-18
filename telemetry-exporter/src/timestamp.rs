//! Millisecond epoch times and their ISO 8601 form.
//!
//! The record schema writes every time as `YYYY-MM-DDTHH:MM:SS.sssZ`, the
//! form JavaScript's `Date.prototype.toISOString` produces. The crate
//! formats it itself, so it needs no date-time dependency.

use std::time::SystemTime;

/// Milliseconds since the Unix epoch. A time before the epoch is negative.
/// A time outside the `i64` range saturates.
pub(crate) fn epoch_ms(time: SystemTime) -> i64 {
    match time.duration_since(SystemTime::UNIX_EPOCH) {
        Ok(after) => i64::try_from(after.as_millis()).unwrap_or(i64::MAX),
        Err(before) => {
            let ms = before.duration().as_millis();
            // A time before the epoch rounds toward the earlier millisecond,
            // as `Date.getTime` floors a time.
            let ms = if before.duration().subsec_nanos() % 1_000_000 == 0 {
                ms
            } else {
                ms.saturating_add(1)
            };
            i64::try_from(ms).map_or(i64::MIN, |ms| -ms)
        }
    }
}

/// The current time in milliseconds since the Unix epoch.
pub(crate) fn now_ms() -> i64 {
    epoch_ms(SystemTime::now())
}

/// The `SystemTime` of an epoch millisecond count.
#[cfg(test)]
pub(crate) fn system_time(ms: i64) -> SystemTime {
    let magnitude = std::time::Duration::from_millis(ms.unsigned_abs());
    if ms >= 0 {
        SystemTime::UNIX_EPOCH + magnitude
    } else {
        SystemTime::UNIX_EPOCH - magnitude
    }
}

/// The calendar date and time of an epoch millisecond count, in UTC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Civil {
    pub(crate) year: i64,
    pub(crate) month: i64,
    pub(crate) day: i64,
    pub(crate) hour: i64,
    pub(crate) minute: i64,
    pub(crate) second: i64,
    pub(crate) millisecond: i64,
}

const MS_PER_DAY: i64 = 86_400_000;

impl Civil {
    /// Splits an epoch millisecond count into UTC calendar fields.
    ///
    /// The date conversion is Howard Hinnant's `civil_from_days`, which is
    /// exact for every day of the proleptic Gregorian calendar.
    pub(crate) fn from_epoch_ms(ms: i64) -> Self {
        let days = ms.div_euclid(MS_PER_DAY);
        let of_day = ms.rem_euclid(MS_PER_DAY);

        let z = days + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z.rem_euclid(146_097);
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = doy - (153 * mp + 2) / 5 + 1;
        let month = if mp < 10 { mp + 3 } else { mp - 9 };
        let year = yoe + era * 400 + i64::from(month <= 2);

        Self {
            year,
            month,
            day,
            hour: of_day / 3_600_000,
            minute: of_day / 60_000 % 60,
            second: of_day / 1000 % 60,
            millisecond: of_day % 1000,
        }
    }
}

/// Formats an epoch millisecond count as `YYYY-MM-DDTHH:MM:SS.sssZ` in UTC.
///
/// A year outside 0..=9999 is written with as many digits as it needs, and
/// a negative year with a leading `-`. JavaScript writes those years in a
/// six-digit expanded form instead. No execution timestamp reaches either
/// range.
pub(crate) fn format_iso_ms(ms: i64) -> String {
    let c = Civil::from_epoch_ms(ms);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        c.year, c.month, c.day, c.hour, c.minute, c.second, c.millisecond
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn formats_the_epoch() {
        assert_eq!(format_iso_ms(0), "1970-01-01T00:00:00.000Z");
    }

    #[test]
    fn formats_a_known_instant_with_milliseconds() {
        // 2026-11-17T08:09:10.123Z, computed with JavaScript's Date.UTC.
        assert_eq!(format_iso_ms(1_794_902_950_123), "2026-11-17T08:09:10.123Z");
    }

    #[test]
    fn formats_a_leap_day() {
        // Date.UTC(2024, 1, 29, 23, 59, 59, 999)
        assert_eq!(format_iso_ms(1_709_251_199_999), "2024-02-29T23:59:59.999Z");
    }

    #[test]
    fn formats_a_time_before_the_epoch() {
        assert_eq!(format_iso_ms(-1), "1969-12-31T23:59:59.999Z");
    }

    #[test]
    fn formats_a_year_past_9999() {
        // Date.UTC(10000, 0, 1)
        assert_eq!(
            format_iso_ms(253_402_300_800_000),
            "10000-01-01T00:00:00.000Z"
        );
    }

    #[test]
    fn epoch_ms_round_trips_through_system_time() {
        for ms in [0, 1, 1_794_902_950_123, -1, -1_500] {
            assert_eq!(epoch_ms(system_time(ms)), ms);
        }
    }

    #[test]
    fn epoch_ms_floors_a_fractional_time_before_the_epoch() {
        let time = SystemTime::UNIX_EPOCH - Duration::from_micros(1_500);
        assert_eq!(epoch_ms(time), -2);
    }

    #[test]
    fn epoch_ms_truncates_a_fractional_time_after_the_epoch() {
        let time = SystemTime::UNIX_EPOCH + Duration::from_micros(1_500);
        assert_eq!(epoch_ms(time), 1);
    }

    #[test]
    fn epoch_ms_saturates_a_time_past_the_i64_range() {
        // `SystemTime` counts seconds in an `i64`, so it holds times whose
        // millisecond count is outside the `i64` range.
        let far = Duration::from_secs(u64::MAX / 2);
        assert_eq!(epoch_ms(SystemTime::UNIX_EPOCH + far), i64::MAX);
        assert_eq!(epoch_ms(SystemTime::UNIX_EPOCH - far), i64::MIN);
    }

    #[test]
    fn now_is_after_2020() {
        assert!(now_ms() > 1_577_836_800_000);
    }
}
