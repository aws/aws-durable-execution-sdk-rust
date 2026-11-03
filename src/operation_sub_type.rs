//! The subtype an operation records in its checkpoint.

use std::borrow::Cow;
use std::fmt;
use std::str::FromStr;

/// The longest subtype the service accepts, in bytes.
const MAX_LEN: usize = 32;

/// An operation subtype: 1 to 32 characters from `A-Z`, `a-z`, `0-9`, `-`
/// and `_`.
///
/// Every operation records a subtype in its checkpoint, such as `Step` or
/// `Callback`. Execution history and plugins read it. A step or a
/// callback can record a custom subtype instead, so the parts of a composed
/// operation can be told apart. Not calling a builder's `sub_type` setter
/// selects the default subtype. The associated constants hold the subtypes
/// the SDK records for its own operations.
///
/// [`new`](Self::new) validates a value against the rule the service
/// applies. The service rejects any other subtype, and the execution then
/// fails. An empty string is rejected too: a caller selects the default by
/// not calling the setter.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk::OperationSubType;
///
/// let label = String::from("Payment");
/// let sub_type = OperationSubType::new(label.as_str())?;
/// assert_eq!(sub_type.as_str(), "Payment");
///
/// assert!(OperationSubType::new("").is_err());
/// assert_eq!(OperationSubType::MAP_ITERATION.as_str(), "MapIteration");
/// # Ok::<(), aws_durable_execution_sdk::InvalidOperationSubType>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct OperationSubType(Cow<'static, str>);

impl OperationSubType {
    /// `Step`: a step.
    pub const STEP: OperationSubType = Self(Cow::Borrowed(crate::step::STEP_SUB_TYPE));
    /// `Wait`: a wait.
    pub const WAIT: OperationSubType = Self(Cow::Borrowed(crate::wait::WAIT_SUB_TYPE));
    /// `Callback`: a callback.
    pub const CALLBACK: OperationSubType = Self(Cow::Borrowed(crate::callback::CALLBACK_SUB_TYPE));
    /// `RunInChildContext`: a child context.
    pub const RUN_IN_CHILD_CONTEXT: OperationSubType =
        Self(Cow::Borrowed(crate::child::CHILD_SUB_TYPE));
    /// `Map`: the parent of a map.
    pub const MAP: OperationSubType = Self(Cow::Borrowed(crate::map_parallel::MAP_SUB_TYPE));
    /// `MapIteration`: one item of a map.
    pub const MAP_ITERATION: OperationSubType =
        Self(Cow::Borrowed(crate::map_parallel::MAP_ITERATION_SUB_TYPE));
    /// `Parallel`: the parent of a parallel.
    pub const PARALLEL: OperationSubType =
        Self(Cow::Borrowed(crate::map_parallel::PARALLEL_SUB_TYPE));
    /// `ParallelBranch`: one branch of a parallel.
    pub const PARALLEL_BRANCH: OperationSubType =
        Self(Cow::Borrowed(crate::map_parallel::PARALLEL_BRANCH_SUB_TYPE));
    /// `WaitForCallback`: the child context of a `wait_for_callback`.
    pub const WAIT_FOR_CALLBACK: OperationSubType =
        Self(Cow::Borrowed(crate::callback::WFCB_SUB_TYPE));
    /// `WaitForCondition`: a `wait_for_condition`.
    pub const WAIT_FOR_CONDITION: OperationSubType =
        Self(Cow::Borrowed(crate::wait_for_condition::WFC_SUB_TYPE));
    /// `ChainedInvoke`: an invoke.
    pub const CHAINED_INVOKE: OperationSubType =
        Self(Cow::Borrowed(crate::invoke::CHAINED_INVOKE_SUB_TYPE));

    /// Validates `value` against the service rule.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidOperationSubType`] unless `value` has 1 to 32
    /// bytes and every byte is an ASCII letter, an ASCII digit, `-` or
    /// `_`.
    pub fn new(value: impl Into<String>) -> Result<OperationSubType, InvalidOperationSubType> {
        let value = value.into();
        let valid = (1..=MAX_LEN).contains(&value.len())
            && value
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_');
        if valid {
            Ok(Self(Cow::Owned(value)))
        } else {
            Err(InvalidOperationSubType { value })
        }
    }

    /// Returns the subtype as a string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for OperationSubType {
    type Err = InvalidOperationSubType;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

impl TryFrom<&str> for OperationSubType {
    type Error = InvalidOperationSubType;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl TryFrom<String> for OperationSubType {
    type Error = InvalidOperationSubType;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl AsRef<str> for OperationSubType {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for OperationSubType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The error [`OperationSubType::new`] returns for a value the service
/// would reject.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct InvalidOperationSubType {
    value: String,
}

impl InvalidOperationSubType {
    /// The rejected value.
    pub fn value(&self) -> &str {
        &self.value
    }
}

impl fmt::Display for InvalidOperationSubType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid subtype {:?}: must be 1 to 32 characters from A-Z, a-z, 0-9, hyphen and underscore",
            self.value
        )
    }
}

impl std::error::Error for InvalidOperationSubType {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows 1 to 9 of the case table, through `new` with a `&str` built at
    /// run time, plus the three conversion impls for rows 1 and 4.
    #[test]
    fn sub_type_new_cases() {
        let a32 = "a".repeat(32);
        let a33 = "a".repeat(33);
        let ok: [(String, Option<OperationSubType>); 3] = [
            ("Payment-1_x".to_owned(), None),
            (a32, None),
            ("Step".to_owned(), Some(OperationSubType::STEP)),
        ];
        for (value, expected) in ok {
            let parsed = OperationSubType::new(value.as_str()).expect("a valid value is accepted");
            assert_eq!(parsed.as_str(), value);
            if let Some(expected) = expected {
                assert_eq!(parsed, expected);
            }
        }

        let bad = [
            String::new(),
            a33,
            "a b".to_owned(),
            "a.b".to_owned(),
            "é".to_owned(),
            "ok\n".to_owned(),
        ];
        for value in bad {
            let err = OperationSubType::new(value.as_str())
                .expect_err(&format!("{value:?} must be rejected"));
            assert_eq!(err.value(), value);
        }

        let row1 = String::from("Payment-1_x");
        let expected = OperationSubType::new(row1.as_str()).unwrap();
        assert_eq!(
            row1.as_str().parse::<OperationSubType>(),
            Ok(expected.clone())
        );
        assert_eq!(
            OperationSubType::try_from(row1.as_str()),
            Ok(expected.clone())
        );
        assert_eq!(OperationSubType::try_from(row1.clone()), Ok(expected));

        let row4 = String::new();
        assert_eq!(
            row4.as_str()
                .parse::<OperationSubType>()
                .unwrap_err()
                .value(),
            ""
        );
        assert_eq!(
            OperationSubType::try_from(row4.as_str())
                .unwrap_err()
                .value(),
            ""
        );
        assert_eq!(OperationSubType::try_from(row4).unwrap_err().value(), "");
    }

    #[test]
    fn built_in_constants_match_js_names() {
        let cases = [
            (OperationSubType::STEP, "Step"),
            (OperationSubType::WAIT, "Wait"),
            (OperationSubType::CALLBACK, "Callback"),
            (OperationSubType::RUN_IN_CHILD_CONTEXT, "RunInChildContext"),
            (OperationSubType::MAP, "Map"),
            (OperationSubType::MAP_ITERATION, "MapIteration"),
            (OperationSubType::PARALLEL, "Parallel"),
            (OperationSubType::PARALLEL_BRANCH, "ParallelBranch"),
            (OperationSubType::WAIT_FOR_CALLBACK, "WaitForCallback"),
            (OperationSubType::WAIT_FOR_CONDITION, "WaitForCondition"),
            (OperationSubType::CHAINED_INVOKE, "ChainedInvoke"),
        ];
        for (constant, name) in cases {
            assert_eq!(constant.as_str(), name);
            assert_eq!(constant.as_ref(), name);
            assert_eq!(constant.to_string(), name);
            assert_eq!(OperationSubType::new(name), Ok(constant));
        }
    }

    #[test]
    fn invalid_sub_type_display_names_the_rule() {
        let err = OperationSubType::new("a b").unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid subtype \"a b\": must be 1 to 32 characters from A-Z, a-z, 0-9, hyphen and underscore"
        );
        let err = OperationSubType::new("ok\n").unwrap_err();
        assert!(
            err.to_string().starts_with("invalid subtype \"ok\\n\": "),
            "{err}"
        );
    }
}
