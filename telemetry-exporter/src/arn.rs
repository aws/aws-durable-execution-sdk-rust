//! The identity fields of an execution ARN, and the sampling decision.

/// The fields of a durable execution ARN that a record carries.
///
/// The ARN format is
/// `arn:<partition>:lambda:<region>:<account>:function:<function>:<qualifier>/durable-execution/<execution name>/<invocation id>`.
/// A missing part is the empty string, as in the JavaScript exporter.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ParsedArn {
    pub(crate) region: String,
    pub(crate) account_id: String,
    pub(crate) function_name: String,
    pub(crate) qualifier: String,
    pub(crate) execution_name: String,
}

impl ParsedArn {
    /// Splits an execution ARN on `:`, then its eighth part on `/`.
    pub(crate) fn parse(arn: &str) -> Self {
        let parts: Vec<&str> = arn.split(':').collect();
        let part = |index: usize| parts.get(index).copied().unwrap_or_default().to_owned();
        let last = parts.get(7).copied().unwrap_or_default();
        let segments: Vec<&str> = last.split('/').collect();
        let segment = |index: usize| segments.get(index).copied().unwrap_or_default().to_owned();
        Self {
            region: part(3),
            account_id: part(4),
            function_name: part(6),
            qualifier: segment(0),
            execution_name: segment(2),
        }
    }
}

/// The 32-bit FNV-1a hash of a string's UTF-16 code units.
///
/// The JavaScript exporter hashes `charCodeAt` values, which are UTF-16
/// code units. Hashing the same units gives the same decision for every
/// ARN, including one with a character outside ASCII.
fn fnv1a32(input: &str) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for unit in input.encode_utf16() {
        hash ^= u32::from(unit);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

/// Whether the execution is sampled in.
///
/// The ARN is the same in every invocation of an execution. So every
/// invocation reaches the same decision. The hash maps into `[0, 1]`, and
/// the execution is sampled in when the result is below `rate`.
pub(crate) fn should_sample(execution_arn: &str, rate: f64) -> bool {
    if rate >= 1.0 {
        return true;
    }
    if rate <= 0.0 {
        return false;
    }
    f64::from(fnv1a32(execution_arn)) / f64::from(u32::MAX) < rate
}

/// Normalizes a configured sampling rate into `[0, 1]`.
///
/// A configuration error must not stop the telemetry. So a NaN rate becomes
/// `1.0` and a rate outside `[0, 1]` is clamped. Each case logs a warning.
pub(crate) fn resolve_sampling_rate(rate: f64) -> f64 {
    if rate.is_nan() {
        tracing::warn!(
            target: crate::TARGET,
            "samplingRate is not a number; defaulting to 1.0 (all executions)"
        );
        return 1.0;
    }
    if !(0.0..=1.0).contains(&rate) {
        let clamped = rate.clamp(0.0, 1.0);
        tracing::warn!(
            target: crate::TARGET,
            "samplingRate {rate} is out of range [0, 1]; clamping to {clamped}"
        );
        return clamped;
    }
    rate
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARN: &str = "arn:aws:lambda:us-west-2:123456789012:function:my-fn:$LATEST/durable-execution/exec-1/inv-1";

    #[test]
    fn parses_every_field() {
        let parsed = ParsedArn::parse(ARN);
        assert_eq!(
            parsed,
            ParsedArn {
                region: "us-west-2".to_owned(),
                account_id: "123456789012".to_owned(),
                function_name: "my-fn".to_owned(),
                qualifier: "$LATEST".to_owned(),
                execution_name: "exec-1".to_owned(),
            }
        );
    }

    #[test]
    fn a_short_arn_leaves_the_missing_parts_empty() {
        let parsed = ParsedArn::parse("arn:aws:lambda:us-west-2:000000000000:function:local-test");
        assert_eq!(parsed.function_name, "local-test");
        assert_eq!(parsed.qualifier, "");
        assert_eq!(parsed.execution_name, "");
    }

    #[test]
    fn an_empty_arn_parses_to_empty_fields() {
        assert_eq!(ParsedArn::parse(""), ParsedArn::default());
    }

    #[test]
    fn fnv1a32_matches_the_reference_vectors() {
        // Published FNV-1a 32-bit test vectors.
        assert_eq!(fnv1a32(""), 0x811c_9dc5);
        assert_eq!(fnv1a32("a"), 0xe40c_292c);
        assert_eq!(fnv1a32("foobar"), 0xbf9c_f968);
    }

    #[test]
    fn fnv1a32_hashes_utf16_code_units() {
        // U+00E9 is one UTF-16 unit (0xE9) but two UTF-8 bytes. The hash
        // of the unit differs from the hash of the bytes.
        let mut by_unit: u32 = 0x811c_9dc5;
        by_unit ^= 0xe9;
        by_unit = by_unit.wrapping_mul(0x0100_0193);
        assert_eq!(fnv1a32("\u{e9}"), by_unit);
    }

    #[test]
    fn a_rate_of_one_or_more_samples_everything() {
        assert!(should_sample(ARN, 1.0));
        assert!(should_sample(ARN, 2.0));
    }

    #[test]
    fn a_rate_of_zero_or_less_samples_nothing() {
        assert!(!should_sample(ARN, 0.0));
        assert!(!should_sample(ARN, -1.0));
    }

    #[test]
    fn a_fractional_rate_compares_the_scaled_hash() {
        let scaled = f64::from(fnv1a32(ARN)) / f64::from(u32::MAX);
        assert!(should_sample(ARN, scaled + 1e-9));
        assert!(!should_sample(ARN, scaled));
    }

    #[test]
    fn the_decision_is_stable_for_one_arn() {
        let first = should_sample(ARN, 0.5);
        for _ in 0..10 {
            assert_eq!(should_sample(ARN, 0.5), first);
        }
    }

    #[test]
    fn resolve_keeps_an_in_range_rate() {
        assert!((resolve_sampling_rate(0.25) - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn resolve_clamps_an_out_of_range_rate() {
        assert!((resolve_sampling_rate(1.5) - 1.0).abs() < f64::EPSILON);
        assert!(resolve_sampling_rate(-0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn resolve_defaults_nan_to_one() {
        assert!((resolve_sampling_rate(f64::NAN) - 1.0).abs() < f64::EPSILON);
    }
}
