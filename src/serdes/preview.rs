//! Field-selecting previews of JSON values, for the `preview` key of a
//! [`FileSystemSerdes`](super::FileSystemSerdes) file envelope.
//!
//! [`build_preview`] ports the JS SDK's `buildPreview`. It works on JSON
//! text, so no `serde_json` type appears in the public API. A private
//! ordered tree keeps object keys in document order without enabling
//! `serde_json`'s `preserve_order` feature.

use std::fmt;

use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};

/// Which fields a preview shows before `include`, `exclude` and `mask`
/// apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PreviewMode {
    /// Start with every field visible; apply `exclude` and `mask`.
    IncludeAll,
    /// Start with no field visible; apply `include` and `mask`.
    ExcludeAll,
}

/// How a [`PreviewField`] matches a field path.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Selector {
    /// Any `.`-separated segment of the path equals the name.
    Anywhere(String),
    /// The whole dotted path equals this path.
    Path(String),
}

/// A field selector: a name matched at any depth, or an exact dotted path.
///
/// A `&str` converts to [`PreviewField::anywhere`].
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk::serdes::PreviewField;
///
/// let by_name = PreviewField::anywhere("email");
/// let by_path = PreviewField::path("user.email");
/// assert_eq!(PreviewField::from("email"), by_name);
/// assert_ne!(by_name, by_path);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct PreviewField {
    selector: Selector,
}

impl PreviewField {
    /// Matches a field with this name at any depth.
    #[must_use]
    pub fn anywhere(name: impl Into<String>) -> Self {
        Self {
            selector: Selector::Anywhere(name.into()),
        }
    }

    /// Matches only the field at this dotted path from the root.
    #[must_use]
    pub fn path(path: impl Into<String>) -> Self {
        Self {
            selector: Selector::Path(path.into()),
        }
    }

    fn matches(&self, path: &str) -> bool {
        match &self.selector {
            Selector::Anywhere(name) => path.split('.').any(|segment| segment == name),
            Selector::Path(exact) => path == exact,
        }
    }
}

impl From<&str> for PreviewField {
    fn from(name: &str) -> Self {
        Self::anywhere(name)
    }
}

/// Configuration for [`build_preview`].
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk::serdes::{PreviewConfig, PreviewMode};
///
/// let config = PreviewConfig::builder(PreviewMode::ExcludeAll)
///     .include("id")
///     .mask("email")
///     .build();
/// # drop(config);
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PreviewConfig {
    mode: PreviewMode,
    include: Vec<PreviewField>,
    exclude: Vec<PreviewField>,
    mask: Vec<PreviewField>,
    mask_string: String,
    max_preview_bytes: usize,
}

impl PreviewConfig {
    /// Creates a builder for a preview in `mode`.
    pub fn builder(mode: PreviewMode) -> PreviewConfigBuilder {
        PreviewConfigBuilder {
            config: Self {
                mode,
                include: Vec::new(),
                exclude: Vec::new(),
                mask: Vec::new(),
                mask_string: "***".to_owned(),
                max_preview_bytes: 4096,
            },
        }
    }
}

/// Builder for [`PreviewConfig`].
#[derive(Debug, Clone)]
#[must_use = "builders do nothing unless .build() is called"]
#[non_exhaustive]
pub struct PreviewConfigBuilder {
    config: PreviewConfig,
}

impl PreviewConfigBuilder {
    /// Adds a field to show. It matters in [`PreviewMode::ExcludeAll`].
    pub fn include(mut self, field: impl Into<PreviewField>) -> Self {
        self.config.include.push(field.into());
        self
    }

    /// Adds a field to hide. An excluded field is never shown, even when it
    /// is also masked.
    pub fn exclude(mut self, field: impl Into<PreviewField>) -> Self {
        self.config.exclude.push(field.into());
        self
    }

    /// Adds a field to show with the mask string instead of its value. A
    /// masked field is shown in either mode unless it is excluded.
    pub fn mask(mut self, field: impl Into<PreviewField>) -> Self {
        self.config.mask.push(field.into());
        self
    }

    /// Sets the text a masked field shows. Default `"***"`.
    pub fn mask_string(mut self, mask: impl Into<String>) -> Self {
        self.config.mask_string = mask.into();
        self
    }

    /// Sets the size limit of the preview in bytes. Default 4096.
    pub fn max_preview_bytes(mut self, bytes: usize) -> Self {
        self.config.max_preview_bytes = bytes;
        self
    }

    /// Builds the config.
    #[must_use]
    pub fn build(self) -> PreviewConfig {
        self.config
    }
}

/// Builds a preview of `json` (a JSON value as text) by `config`.
/// Returns the preview as compact JSON object text, or `None`.
///
/// The walk follows the JS SDK's `buildPreview`. A field is visible when it
/// is not excluded and is masked, or is included, or the mode is
/// [`PreviewMode::IncludeAll`]. A visible masked field shows the mask
/// string. A visible object or array, and a field that is neither visible
/// nor excluded, is walked into. The fields collected this way are kept in
/// order until the next one would make the preview larger than the size
/// limit, and are rebuilt into one nested object.
///
/// `None` comes back when `json` does not parse, when its root is neither
/// an object nor an array, or when no field fits the selection and the
/// size limit.
///
/// # Limitations
///
/// As in the JS SDK:
///
/// - Keys that contain `.` are skipped, because a path cannot tell them
///   from nested keys. So are the keys `__proto__`, `constructor` and
///   `prototype`.
/// - Arrays are merged into one object: the fields of every element land
///   at the same path, and a later element overwrites an earlier one.
/// - Arrays of primitives are dropped.
/// - Keys are visited in document order, so the size limit drops trailing
///   fields.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk::serdes::{PreviewConfig, PreviewMode, build_preview};
///
/// let config = PreviewConfig::builder(PreviewMode::ExcludeAll)
///     .include("id")
///     .include("status")
///     .mask("email")
///     .build();
/// let preview = build_preview(
///     r#"{"id":"order-123","status":"pending","user":{"email":"a@example.com","name":"Bo"},"total":5}"#,
///     &config,
/// );
/// assert_eq!(
///     preview.as_deref(),
///     Some(r#"{"id":"order-123","status":"pending","user":{"email":"***"}}"#),
/// );
/// ```
#[must_use]
pub fn build_preview(json: &str, config: &PreviewConfig) -> Option<String> {
    let root = parse_ordered(json)?;
    if !matches!(root, Node::Object(_) | Node::Array(_)) {
        return None;
    }

    let mut pairs = Vec::new();
    collect(&root, "", config, &mut pairs);

    // "{}"
    let mut estimated_size = 2;
    let mut result = Vec::new();
    for (path, value) in pairs {
        // The UTF-8 length of `"<path>":<value>,`, as JS estimates it.
        let entry_size = path.len() + render_compact(&value).len() + 4;
        if estimated_size + entry_size > config.max_preview_bytes {
            break;
        }
        estimated_size += entry_size;
        assign(&mut result, &path, value);
    }

    if result.is_empty() {
        None
    } else {
        Some(render_compact(&Node::Object(result)))
    }
}

/// Keys JS skips so that a preview can never set an object's prototype.
const DANGEROUS_KEYS: [&str; 3] = ["__proto__", "constructor", "prototype"];

/// Collects `(path, value)` pairs from `node` in traversal order.
fn collect(node: &Node, prefix: &str, config: &PreviewConfig, pairs: &mut Vec<(String, Node)>) {
    match node {
        Node::Array(items) => {
            for item in items {
                collect(item, prefix, config, pairs);
            }
        }
        Node::Object(entries) => {
            for (key, value) in entries {
                if DANGEROUS_KEYS.contains(&key.as_str()) || key.contains('.') {
                    continue;
                }
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                let masked = config.mask.iter().any(|f| f.matches(&path));
                let excluded = config.exclude.iter().any(|f| f.matches(&path));
                let visible = !excluded
                    && (masked
                        || config.mode == PreviewMode::IncludeAll
                        || config.include.iter().any(|f| f.matches(&path)));

                if !visible {
                    if !excluded {
                        collect(value, &path, config, pairs);
                    }
                } else if masked {
                    pairs.push((path, Node::String(config.mask_string.clone())));
                } else if matches!(value, Node::Object(_) | Node::Array(_)) {
                    collect(value, &path, config, pairs);
                } else {
                    pairs.push((path, value.clone()));
                }
            }
        }
        // `collect` is called on primitives when a hidden or nested field
        // holds one; JS returns without collecting anything.
        Node::Null | Node::Bool(_) | Node::Number(_) | Node::String(_) => {}
    }
}

/// Sets `path` to `value` in `object`, creating intermediate objects.
///
/// An existing key keeps its position. An intermediate key that holds no
/// object is set to an empty object, as JS's assignment does.
fn assign(object: &mut Vec<(String, Node)>, path: &str, value: Node) {
    let (key, rest) = match path.split_once('.') {
        Some((key, rest)) => (key, Some(rest)),
        None => (path, None),
    };
    for (existing_key, slot) in object.iter_mut() {
        if existing_key == key {
            let existing = std::mem::replace(slot, Node::Null);
            *slot = assigned(existing, rest, value);
            return;
        }
    }
    object.push((key.to_owned(), assigned(Node::Null, rest, value)));
}

/// The new value of a key that held `existing`, after setting `rest`
/// below it to `value`.
fn assigned(existing: Node, rest: Option<&str>, value: Node) -> Node {
    match rest {
        None => value,
        Some(path) => {
            let mut children = match existing {
                Node::Object(children) => children,
                _ => Vec::new(),
            };
            assign(&mut children, path, value);
            Node::Object(children)
        }
    }
}

/// A JSON value whose objects keep their keys in document order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Node {
    Null,
    Bool(bool),
    /// The number's compact JSON text.
    Number(String),
    String(String),
    Array(Vec<Node>),
    Object(Vec<(String, Node)>),
}

/// Parses `json` into an ordered tree, or `None` when it is not JSON.
///
/// A key that appears twice in one object keeps its first position and
/// takes its last value, as JS's `JSON.parse` does.
pub(crate) fn parse_ordered(json: &str) -> Option<Node> {
    serde_json::from_str(json).ok()
}

/// Renders `node` as compact JSON text, keeping its key order.
pub(crate) fn render_compact(node: &Node) -> String {
    let mut out = String::new();
    render_into(node, &mut out);
    out
}

fn render_into(node: &Node, out: &mut String) {
    match node {
        Node::Null => out.push_str("null"),
        Node::Bool(true) => out.push_str("true"),
        Node::Bool(false) => out.push_str("false"),
        Node::Number(text) => out.push_str(text),
        Node::String(text) => render_string(text, out),
        Node::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                render_into(item, out);
            }
            out.push(']');
        }
        Node::Object(entries) => {
            out.push('{');
            for (index, (key, value)) in entries.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                render_string(key, out);
                out.push(':');
                render_into(value, out);
            }
            out.push('}');
        }
    }
}

fn render_string(text: &str, out: &mut String) {
    out.push_str(&serde_json::Value::String(text.to_owned()).to_string());
}

impl<'de> Deserialize<'de> for Node {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(NodeVisitor)
    }
}

struct NodeVisitor;

impl<'de> Visitor<'de> for NodeVisitor {
    type Value = Node;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a JSON value")
    }

    fn visit_unit<E: de::Error>(self) -> Result<Node, E> {
        Ok(Node::Null)
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> Result<Node, E> {
        Ok(Node::Bool(value))
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Node, E> {
        Ok(Node::Number(value.to_string()))
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Node, E> {
        Ok(Node::Number(value.to_string()))
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Node, E> {
        Ok(Node::Number(serde_json::Value::from(value).to_string()))
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Node, E> {
        Ok(Node::String(value.to_owned()))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Node, A::Error> {
        let mut items = Vec::new();
        while let Some(item) = seq.next_element()? {
            items.push(item);
        }
        Ok(Node::Array(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Node, A::Error> {
        let mut entries: Vec<(String, Node)> = Vec::new();
        while let Some((key, value)) = map.next_entry::<String, Node>()? {
            match entries.iter_mut().find(|(existing, _)| *existing == key) {
                Some((_, slot)) => *slot = value,
                None => entries.push((key, value)),
            }
        }
        Ok(Node::Object(entries))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const V: &str = r#"{"id":"123","email":"alice@example.com","ssn":"000-00-0000","user":{"name":"Alice","role":"admin"}}"#;

    fn parsed(text: &str) -> serde_json::Value {
        serde_json::from_str(text).unwrap()
    }

    fn preview(json: &str, config: &PreviewConfig) -> Option<serde_json::Value> {
        build_preview(json, config).map(|text| parsed(&text))
    }

    fn v_with(
        edit: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>),
    ) -> serde_json::Value {
        let mut value = parsed(V);
        if let serde_json::Value::Object(map) = &mut value {
            edit(map);
        }
        value
    }

    // 1
    #[test]
    fn include_all_keeps_every_field() {
        let config = PreviewConfig::builder(PreviewMode::IncludeAll).build();
        assert_eq!(preview(V, &config), Some(parsed(V)));
    }

    // 2
    #[test]
    fn include_all_with_exclude_drops_the_field() {
        let config = PreviewConfig::builder(PreviewMode::IncludeAll)
            .exclude("ssn")
            .build();
        let expected = v_with(|map| {
            map.remove("ssn");
        });
        assert_eq!(preview(V, &config), Some(expected));
    }

    // 3
    #[test]
    fn exclude_all_with_include_keeps_only_included() {
        let config = PreviewConfig::builder(PreviewMode::ExcludeAll)
            .include("id")
            .include("email")
            .build();
        assert_eq!(
            build_preview(V, &config).as_deref(),
            Some(r#"{"id":"123","email":"alice@example.com"}"#)
        );
    }

    // 4
    #[test]
    fn mask_replaces_the_value_with_the_default_mask() {
        let config = PreviewConfig::builder(PreviewMode::IncludeAll)
            .mask("ssn")
            .build();
        let expected = v_with(|map| {
            map.insert("ssn".to_owned(), "***".into());
        });
        assert_eq!(preview(V, &config), Some(expected));
    }

    // 5
    #[test]
    fn mask_applies_inside_array_elements() {
        let config = PreviewConfig::builder(PreviewMode::IncludeAll)
            .mask("secret")
            .build();
        assert_eq!(
            preview(r#"{"items":[{"secret":"xyz"},{"secret":"abc"}]}"#, &config),
            Some(parsed(r#"{"items":{"secret":"***"}}"#))
        );
    }

    // 6
    #[test]
    fn custom_mask_string_is_used() {
        let config = PreviewConfig::builder(PreviewMode::IncludeAll)
            .mask("ssn")
            .mask_string("[REDACTED]")
            .build();
        let expected = v_with(|map| {
            map.insert("ssn".to_owned(), "[REDACTED]".into());
        });
        assert_eq!(preview(V, &config), Some(expected));
    }

    const EMAILS: &str = r#"{"email":"root@example.com","user":{"email":"nested@example.com"}}"#;

    // 7
    #[test]
    fn path_selector_matches_only_the_root_field() {
        let config = PreviewConfig::builder(PreviewMode::ExcludeAll)
            .include(PreviewField::path("email"))
            .build();
        assert_eq!(
            preview(EMAILS, &config),
            Some(parsed(r#"{"email":"root@example.com"}"#))
        );
    }

    // 8
    #[test]
    fn anywhere_selector_matches_at_every_depth() {
        let config = PreviewConfig::builder(PreviewMode::ExcludeAll)
            .include("email")
            .build();
        assert_eq!(preview(EMAILS, &config), Some(parsed(EMAILS)));
    }

    // 9
    #[test]
    fn mask_makes_a_field_visible_in_exclude_all() {
        let config = PreviewConfig::builder(PreviewMode::ExcludeAll)
            .mask("ssn")
            .build();
        assert_eq!(preview(V, &config), Some(parsed(r#"{"ssn":"***"}"#)));
    }

    // 10
    #[test]
    fn exclude_wins_over_mask() {
        let config = PreviewConfig::builder(PreviewMode::IncludeAll)
            .exclude("ssn")
            .mask("ssn")
            .build();
        let expected = v_with(|map| {
            map.remove("ssn");
        });
        assert_eq!(preview(V, &config), Some(expected));
    }

    // 11
    #[test]
    fn exclude_all_without_selectors_is_none() {
        let config = PreviewConfig::builder(PreviewMode::ExcludeAll).build();
        assert_eq!(build_preview(V, &config), None);
    }

    // 12
    #[test]
    fn path_exclude_drops_the_nested_subtree() {
        let config = PreviewConfig::builder(PreviewMode::IncludeAll)
            .exclude(PreviewField::path("customer.address"))
            .build();
        assert_eq!(
            preview(
                r#"{"orderId":"o-1","customer":{"address":{"city":"Seattle","zip":"98109"}}}"#,
                &config
            ),
            Some(parsed(r#"{"orderId":"o-1"}"#))
        );
    }

    // 13
    #[test]
    fn deep_path_exclude_keeps_sibling_fields() {
        let config = PreviewConfig::builder(PreviewMode::IncludeAll)
            .exclude(PreviewField::path("a.b.c"))
            .build();
        assert_eq!(
            build_preview(
                r#"{"id":"1","a":{"b":{"c":{"secret":"hidden"},"d":"visible"}}}"#,
                &config
            )
            .as_deref(),
            Some(r#"{"id":"1","a":{"b":{"d":"visible"}}}"#)
        );
    }

    // 14
    #[test]
    fn js_doc_example_matches() {
        let config = PreviewConfig::builder(PreviewMode::ExcludeAll)
            .include("id")
            .include("status")
            .mask("email")
            .build();
        assert_eq!(
            build_preview(
                r#"{"id":"order-123","status":"pending","user":{"email":"a@example.com","name":"Bo"},"total":5}"#,
                &config
            )
            .as_deref(),
            Some(r#"{"id":"order-123","status":"pending","user":{"email":"***"}}"#)
        );
    }

    // 15
    #[test]
    fn size_limit_stops_at_the_first_field_that_does_not_fit() {
        let config = PreviewConfig::builder(PreviewMode::IncludeAll)
            .max_preview_bytes(40)
            .build();
        assert_eq!(preview(V, &config), Some(parsed(r#"{"id":"123"}"#)));
    }

    // 16
    #[test]
    fn size_limit_below_the_first_field_is_none() {
        let config = PreviewConfig::builder(PreviewMode::IncludeAll)
            .max_preview_bytes(12)
            .build();
        assert_eq!(build_preview(V, &config), None);
    }

    // 17
    #[test]
    fn primitive_or_invalid_root_is_none() {
        let config = PreviewConfig::builder(PreviewMode::IncludeAll).build();
        for json in [r#""string""#, "42", "null", "{"] {
            assert_eq!(build_preview(json, &config), None, "{json}");
        }
    }

    // 18
    #[test]
    fn dangerous_and_dotted_keys_are_skipped() {
        let config = PreviewConfig::builder(PreviewMode::IncludeAll).build();
        assert_eq!(
            preview(r#"{"__proto__":1,"a.b":2,"c":3}"#, &config),
            Some(parsed(r#"{"c":3}"#))
        );
        for key in ["constructor", "prototype"] {
            let json = format!(r#"{{"{key}":1,"c":3}}"#);
            assert_eq!(preview(&json, &config), Some(parsed(r#"{"c":3}"#)), "{key}");
        }
    }

    // 19
    #[test]
    fn arrays_of_primitives_are_dropped() {
        let config = PreviewConfig::builder(PreviewMode::IncludeAll).build();
        assert_eq!(
            preview(r#"{"tags":["x","y"],"id":1}"#, &config),
            Some(parsed(r#"{"id":1}"#))
        );
    }

    // 20
    #[test]
    fn root_array_elements_merge_and_the_last_wins() {
        let config = PreviewConfig::builder(PreviewMode::IncludeAll).build();
        assert_eq!(
            preview(r#"[{"id":1},{"id":2}]"#, &config),
            Some(parsed(r#"{"id":2}"#))
        );
    }

    // 21
    #[test]
    fn later_object_replaces_an_earlier_primitive() {
        let config = PreviewConfig::builder(PreviewMode::IncludeAll).build();
        assert_eq!(
            preview(r#"[{"user":"arb"},{"user":{"email":"x"}}]"#, &config),
            Some(parsed(r#"{"user":{"email":"x"}}"#))
        );
    }

    // 22
    #[test]
    fn preview_cut_keeps_document_order() {
        let config = PreviewConfig::builder(PreviewMode::IncludeAll)
            .max_preview_bytes(12)
            .build();
        assert_eq!(
            preview(r#"{"b":"1","a":"2"}"#, &config),
            Some(parsed(r#"{"b":"1"}"#))
        );
    }

    #[test]
    fn later_primitive_replaces_an_earlier_object_in_place() {
        let config = PreviewConfig::builder(PreviewMode::IncludeAll).build();
        assert_eq!(
            build_preview(r#"[{"user":{"email":"x"},"id":1},{"user":"arb"}]"#, &config).as_deref(),
            Some(r#"{"user":"arb","id":1}"#)
        );
    }

    #[test]
    fn parse_ordered_and_render_compact_keep_order_and_values() {
        let text = r#"{ "z": [1, -2, 1.5, true, false, null], "a": "q\"\n\u0001" }"#;
        let node = parse_ordered(text).unwrap();
        assert_eq!(
            render_compact(&node),
            r#"{"z":[1,-2,1.5,true,false,null],"a":"q\"\n\u0001"}"#
        );
        assert_eq!(parse_ordered("{"), None);
    }

    #[test]
    fn duplicate_key_keeps_first_position_and_last_value() {
        let node = parse_ordered(r#"{"a":1,"b":2,"a":3}"#).unwrap();
        assert_eq!(render_compact(&node), r#"{"a":3,"b":2}"#);
    }

    #[test]
    fn escaped_string_value_is_unescaped() {
        let node = parse_ordered(r#""a\tb""#).unwrap();
        assert_eq!(node, Node::String("a\tb".to_owned()));
    }

    #[test]
    fn defaults_are_the_js_defaults() {
        let config = PreviewConfig::builder(PreviewMode::IncludeAll).build();
        assert_eq!(config.mask_string, "***");
        assert_eq!(config.max_preview_bytes, 4096);
    }

    #[test]
    fn preview_field_from_str_is_anywhere() {
        assert_eq!(PreviewField::from("x"), PreviewField::anywhere("x"));
        assert_ne!(PreviewField::anywhere("x"), PreviewField::path("x"));
        assert!(PreviewField::anywhere("b").matches("a.b.c"));
        assert!(!PreviewField::path("b").matches("a.b"));
        assert!(PreviewField::path("a.b").matches("a.b"));
    }
}
