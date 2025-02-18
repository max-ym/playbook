use super::*;


/// Stub for a node. This is a configuration that allows to create a node in the canvas.
/// The same stub can be reused to effectively clone the node.
#[derive(Debug, Clone)]
#[wasm_bindgen(js_name = NodeStub)]
pub struct JsNodeStub {
    pub(crate) stub: canvas::NodeStub,
}

#[wasm_bindgen(js_class = NodeStub)]
impl JsNodeStub {
    /// Get kind of the node stub.
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> JsNodeStubKind {
        use canvas::NodeStub as R;
        use JsNodeStubKind as JS;

        match &self.stub {
            R::File => JS::File,
            R::SplitBy { .. } => JS::SplitBy,
            R::Input { .. } => JS::Input,
            R::Drop => JS::Drop,
            R::Output { .. } => JS::Output,
            R::StrOp(op) => match op {
                canvas::StrOp::Lowercase => JS::Lowercase,
                canvas::StrOp::Uppercase => JS::Uppercase,
                canvas::StrOp::Strip { .. } => JS::Strip,
            },
            R::Compare { .. } => JS::Compare,
            R::Ordering => JS::Ordering,
            R::Regex(_) => JS::Regex,
            R::FindRecord { .. } => JS::FindRecord,
            R::Map { .. } => JS::Map,
            R::List { .. } => JS::List,
            R::IfElse { .. } => JS::IfElse,
            R::OkOrErr => JS::OkOrErr,
            R::Validate { .. } => JS::Validate,
            R::SelectFirst { .. } => JS::SelectFirst,
            R::ExpectOne { .. } => JS::ExpectOne,
            R::ExpectSome { .. } => JS::ExpectSome,
            R::Match { .. } => JS::Match,
            R::Func { .. } => JS::Func,
            R::ParseDateTime { .. } => JS::ParseDateTime,
            R::ParseMonetary { .. } => JS::ParseMonetary,
            R::ParseInt { .. } => JS::ParseInt,
            R::Constant(_) => JS::Constant,
            R::OkOrCrash { .. } => JS::OkOrCrash,
            R::Crash { .. } => JS::Crash,
            R::Unreachable { .. } => JS::Unreachable,
            R::Todo { .. } => JS::Todo,
            R::Comment { .. } => JS::Comment,
        }
    }
}

impl From<canvas::NodeStub> for JsNodeStub {
    fn from(stub: canvas::NodeStub) -> Self {
        Self { stub }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = FileNodeStub)]
pub struct JsFileNodeStub {}

#[wasm_bindgen(js_class = FileNodeStub)]
impl JsFileNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsFileNodeStub {
        JsFileNodeStub {}
    }

    #[wasm_bindgen(getter)]
    pub fn stub(&self) -> JsNodeStub {
        canvas::NodeStub::File.into()
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = SplitByNodeStub)]
pub struct JsSplitByNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub regex: RegExp,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = SplitByNodeStub)]
impl JsSplitByNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn from_regex(js_regex: RegExp) -> JsSplitByNodeStub {
        use lazy_regex::regex::Regex;
        let s = String::from(js_regex.source());
        let regex = Regex::new(&s).expect("it was valid for JS and should be for us as well");

        JsSplitByNodeStub {
            regex: js_regex,
            stub: canvas::NodeStub::SplitBy { regex }.into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = InputNodeStub)]
pub struct JsInputNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone, js_name = validNames)]
    pub valid_names: Vec<JsString>,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = InputNodeStub)]
impl JsInputNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(js_valid_names: Vec<JsString>) -> JsInputNodeStub {
        let valid_names = js_valid_names
            .iter()
            .map(|s| String::from(s).into())
            .collect();
        let stub = canvas::NodeStub::Input { valid_names }.into();
        JsInputNodeStub {
            valid_names: js_valid_names,
            stub,
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = DropNodeStub)]
pub struct JsDropNodeStub {}

#[wasm_bindgen(js_class = DropNodeStub)]
impl JsDropNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsDropNodeStub {
        JsDropNodeStub {}
    }

    #[wasm_bindgen(getter)]
    pub fn stub(&self) -> JsNodeStub {
        canvas::NodeStub::Drop.into()
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = OutputNodeStub)]
pub struct JsOutputNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub ident: JsString,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = OutputNodeStub)]
impl JsOutputNodeStub {
    /// Create a new output node stub.
    ///
    /// # Errors
    /// Passed in string should be valid identifier:
    /// - It should start with a letter or underscore.
    /// - It should contain only letters, digits, and underscores.
    /// - No longer than 64 characters.
    #[wasm_bindgen(constructor)]
    pub fn new(name: JsString) -> Result<JsOutputNodeStub, JsError> {
        let s = String::from(name.clone());
        if s.len() > 64 {
            Err(JsError::new("Name is too long"))
        } else if lazy_regex::regex_is_match!(r"^[a-zA-Z_][a-zA-Z0-9_]*$", &s) {
            let stub = canvas::NodeStub::Output { ident: s.into() }.into();
            Ok(JsOutputNodeStub { ident: name, stub })
        } else {
            Err(JsError::new("Invalid name"))
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = LowercaseNodeStub)]
pub struct JsLowercaseNodeStub {}

#[wasm_bindgen(js_class = LowercaseNodeStub)]
impl JsLowercaseNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsLowercaseNodeStub {
        JsLowercaseNodeStub {}
    }

    #[wasm_bindgen(getter)]
    pub fn stub(&self) -> JsNodeStub {
        use canvas::*;
        NodeStub::StrOp(StrOp::Lowercase).into()
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = UppercaseNodeStub)]
pub struct JsUppercaseNodeStub {}

#[wasm_bindgen(js_class = UppercaseNodeStub)]
impl JsUppercaseNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsUppercaseNodeStub {
        JsUppercaseNodeStub {}
    }

    #[wasm_bindgen(getter)]
    pub fn stub(&self) -> JsNodeStub {
        use canvas::*;
        NodeStub::StrOp(StrOp::Uppercase).into()
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = StripNodeStub)]
pub struct JsStripNodeStub {
    #[wasm_bindgen(readonly, js_name = trimWhitespace)]
    pub trim_whitespace: bool,
    #[wasm_bindgen(readonly, js_name = trimEndWhitespace)]
    pub trim_end_whitespace: bool,
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub remove: Vec<JsString>,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = StripNodeStub)]
impl JsStripNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(
        trim_whitespace: bool,
        trim_end_whitespace: bool,
        remove: Vec<JsString>,
    ) -> JsStripNodeStub {
        use canvas::*;

        let stub = NodeStub::StrOp(StrOp::Strip {
            trim_whitespace,
            trim_end_whitespace,
            remove: remove.iter().map(|s| String::from(s).into()).collect(),
        })
        .into();

        JsStripNodeStub {
            trim_whitespace,
            trim_end_whitespace,
            remove,
            stub,
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = CompareNodeStub)]
pub struct JsCompareNodeStub {
    #[wasm_bindgen(readonly)]
    pub eq: bool,
}

#[wasm_bindgen(js_class = CompareNodeStub)]
impl JsCompareNodeStub {
    pub fn eq() -> JsCompareNodeStub {
        JsCompareNodeStub { eq: true }
    }

    pub fn ne() -> JsCompareNodeStub {
        JsCompareNodeStub { eq: false }
    }

    #[wasm_bindgen(getter)]
    pub fn stub(&self) -> JsNodeStub {
        canvas::NodeStub::Compare { eq: self.eq }.into()
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = OrderingNodeStub)]
pub struct JsOrderingNodeStub;

#[wasm_bindgen(js_class = OrderingNodeStub)]
impl JsOrderingNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn ordering() -> JsOrderingNodeStub {
        JsOrderingNodeStub
    }

    #[wasm_bindgen(getter)]
    pub fn stub(&self) -> JsNodeStub {
        canvas::NodeStub::Ordering.into()
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = RegexNodeStub)]
pub struct JsRegexNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub regex: RegExp,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = RegexNodeStub)]
impl JsRegexNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn from_regex(js_regex: RegExp) -> JsRegexNodeStub {
        let regex = js_regex.into_rust();
        JsRegexNodeStub {
            regex: js_regex,
            stub: canvas::NodeStub::Regex(regex).into(),
        }
    }
}

trait IntoRegex {
    fn into_rust(&self) -> lazy_regex::regex::Regex;
}

impl IntoRegex for RegExp {
    fn into_rust(&self) -> lazy_regex::regex::Regex {
        use lazy_regex::regex::Regex;
        let s = String::from(self.source());
        Regex::new(&s).expect("JS thinks it is valid...")
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = MapNodeStubBuilder)]
pub struct JsMapNodeStubBuilder {
    /// Map given pattern to given value(s). Many values can be given,
    /// if the node has several outputs, corresponding to each value position.
    map: HashMap<canvas::Pat, canvas::Value>,

    #[wasm_bindgen(getter_with_clone)]
    pub wildcard: Vec<JsDataInstance>,
}

#[wasm_bindgen(js_class = MapNodeStubBuilder)]
impl JsMapNodeStubBuilder {
    /// Add "or pattern" to the map. This is a pattern that matches any of the given keys.
    /// The output value will be set to the given value.
    /// If this function was already called for given keys, the value will remain as was set
    /// by the first call.
    #[wasm_bindgen(js_name = orPat)]
    pub fn or_pat(&mut self, keys: Vec<JsDataInstance>, value: JsDataInstance) {
        let pat = canvas::Pat::from_variants(keys.into_iter().map(|v| v.value).collect());
        self.map.entry(pat).or_insert(value.value);
    }

    /// Validate and build the map node stub.
    pub fn build(self) -> Result<JsMapNodeStub, JsError> {
        let wildcard = if self.wildcard.is_empty() {
            None
        } else if self.wildcard.len() == 1 {
            let w = self.wildcard.into_iter().next().unwrap();
            Some(w.value)
        } else {
            Some(canvas::Value::Array(
                self.wildcard.into_iter().map(|v| v.value).collect(),
            ))
        };

        if self.map.is_empty() {
            return Ok(JsMapNodeStub {
                stub: canvas::NodeStub::Map {
                    tuples: Default::default(),
                    wildcard,
                }
                .into(),
            });
        }

        // If keys/values are arrays, all corresponding arrays should be the same size.
        // All values and keys should have the same type.
        let (first_pat, first_val) = self
            .map
            .iter()
            .next()
            .expect("at least one key should exist, per guard above");

        for (pat, val) in self.map.iter().skip(1) {
            if !pat.is_compatible_with(first_pat) {
                return Err(JsError::new("all keys should have the same types"));
            }
            if !val.is_same_type(first_val) {
                return Err(JsError::new("all values should have the same type"));
            }
        }

        Ok(JsMapNodeStub {
            stub: canvas::NodeStub::Map {
                tuples: self.map.into_iter().collect(),
                wildcard,
            }
            .into(),
        })
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = MapNodeStub)]
pub struct JsMapNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = MapNodeStub)]
impl JsMapNodeStub {
    #[wasm_bindgen(getter, js_name = map)]
    pub fn map(&self) -> js_sys::Map {
        if let canvas::NodeStub::Map { tuples, .. } = &self.stub.stub {
            let map = js_sys::Map::new();

            for (pat, value) in tuples.clone() {
                map.set(
                    &JsDataInstance {
                        value: pat.into_value(),
                    }
                    .into(),
                    &JsDataInstance { value }.into(),
                );
            }

            map
        } else {
            unreachable!("stub is not Map, but object is MapNodeStub");
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = ListNodeStub)]
pub struct JsListNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen]
impl JsListNodeStub {
    /// Add element to the list node stub.
    /// This will check if the element is of the same type as the first element in the list.
    /// If the list is empty, the element will be added.
    ///
    /// # Errors
    /// If the element is not of the same type as the first element, this will throw an error.
    #[wasm_bindgen(js_name = addElement)]
    pub fn add_element(&mut self, element: JsDataInstance) -> Result<(), JsInvalidTypeError> {
        if let canvas::NodeStub::List { values } = &mut self.stub.stub {
            if let Some(first) = values.first() {
                if !element.value.is_same_type(first) {
                    return Err(JsInvalidTypeError {
                        got: element.value.type_of().into(),
                        expected: first.type_of().into(),
                    });
                }
            }

            values.push(element.value);
            Ok(())
        } else {
            unreachable!("stub is not List, but object is ListNodeStub");
        }
    }

    /// Get the elements in the list node stub.
    #[wasm_bindgen(getter, js_name = elements)]
    pub fn elements(&self) -> Vec<JsDataInstance> {
        if let canvas::NodeStub::List { values } = &self.stub.stub {
            values
                .iter()
                .map(|v| JsDataInstance { value: v.clone() })
                .collect()
        } else {
            unreachable!("stub is not List, but object is ListNodeStub");
        }
    }

    /// Remove element at the given index. If there is no element at the given index,
    /// this will do nothing.
    #[wasm_bindgen(js_name = removeElementAt)]
    pub fn remove_element_at(&mut self, idx: usize) {
        if let canvas::NodeStub::List { values } = &mut self.stub.stub {
            if idx < values.len() {
                let _ = values.remove(idx);
            }
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = InvalidTypeError)]
pub struct JsInvalidTypeError {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub got: JsDataType,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub expected: JsDataType,
}

#[derive(Debug)]
#[wasm_bindgen(js_name = IfElseNodeStub)]
pub struct JsIfElseNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = IfElseNodeStub)]
impl JsIfElseNodeStub {
    /// Create a new if-else node stub with given amount of data inputs. Note one more first
    /// input pin that is added on top to accept boolean value for the evaluated condition.
    #[wasm_bindgen(constructor)]
    pub fn new(inputs: usize) -> JsIfElseNodeStub {
        JsIfElseNodeStub {
            stub: canvas::NodeStub::IfElse {
                inputs: inputs as canvas::PinOrder,
            }
            .into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = OkOrErrNodeStub)]
pub struct JsOkOrErrNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = OkOrErrNodeStub)]
impl JsOkOrErrNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsOkOrErrNodeStub {
        JsOkOrErrNodeStub {
            stub: canvas::NodeStub::OkOrErr.into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = ValidateNodeStub)]
pub struct JsValidateNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = ValidateNodeStub)]
impl JsValidateNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsValidateNodeStub {
        todo!("ValidateNodeStub")
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = SelectFirstNodeStub)]
pub struct JsSelectFirstNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = SelectFirstNodeStub)]
impl JsSelectFirstNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(inputs: usize) -> JsSelectFirstNodeStub {
        JsSelectFirstNodeStub {
            stub: canvas::NodeStub::SelectFirst {
                inputs: inputs as canvas::PinOrder,
            }
            .into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = ExpectOneNodeStub)]
pub struct JsExpectOneNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = ExpectOne)]
impl JsExpectOneNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(inputs: usize) -> JsExpectOneNodeStub {
        JsExpectOneNodeStub {
            stub: canvas::NodeStub::ExpectOne {
                inputs: inputs as canvas::PinOrder,
            }
            .into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = ExpectSomeNodeStub)]
pub struct JsExpectSomeNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = ExpectSomeNodeStub)]
impl JsExpectSomeNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(msg: String) -> JsExpectSomeNodeStub {
        JsExpectSomeNodeStub {
            stub: canvas::NodeStub::ExpectSome { msg: msg.into() }.into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = MatchNodeStub)]
pub struct JsMatchNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = MatchNode)]
impl JsMatchNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsMatchNodeStub {
        todo!("Similar as Map, DRY code?")
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = FuncNodeStub)]
pub struct JsFuncNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = FuncNodeStub)]
impl JsFuncNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsFuncNodeStub {
        todo!("predicates and functions?")
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = ParseDateTimeNodeStub)]
pub struct JsParseDateTimeNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub format: JsString,

    #[wasm_bindgen(readonly)]
    pub date: bool,

    #[wasm_bindgen(readonly)]
    pub time: bool,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = ParseDateTimeNodeStub)]
impl JsParseDateTimeNodeStub {
    fn new(format: JsString, date: bool, time: bool) -> Self {
        let rust_format = String::from(&format).into();
        JsParseDateTimeNodeStub {
            format,
            date,
            time,
            stub: canvas::NodeStub::ParseDateTime {
                format: rust_format,
                date,
                time,
            }
            .into(),
        }
    }

    #[wasm_bindgen(js_name = date)]
    pub fn date(format: JsString) -> JsParseDateTimeNodeStub {
        Self::new(format, true, false)
    }

    #[wasm_bindgen(js_name = time)]
    pub fn time(format: JsString) -> JsParseDateTimeNodeStub {
        Self::new(format, false, true)
    }

    #[wasm_bindgen(js_name = dateTime)]
    pub fn date_time(format: JsString) -> JsParseDateTimeNodeStub {
        Self::new(format, true, true)
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = ParseMonetaryNodeStub)]
pub struct JsParseMonetaryNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub regex: RegExp,
}

#[wasm_bindgen(js_class = ParseMonetaryNodeStub)]
impl JsParseMonetaryNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(js_regex: RegExp) -> JsParseMonetaryNodeStub {
        let regex = js_regex.into_rust();
        JsParseMonetaryNodeStub {
            regex: js_regex,
            stub: canvas::NodeStub::ParseMonetary { regex }.into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = ParseIntNodeStub)]
pub struct JsParseIntNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,

    #[wasm_bindgen(readonly, js_name = isSigned)]
    pub is_signed: bool,
}

#[wasm_bindgen(js_class = ParseIntNodeStub)]
impl JsParseIntNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(is_signed: bool) -> JsParseIntNodeStub {
        JsParseIntNodeStub {
            is_signed,
            stub: canvas::NodeStub::ParseInt { signed: is_signed }.into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = ConstantNodeStub)]
pub struct JsConstantNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub value: JsDataInstance,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = ConstantNodeStub)]
impl JsConstantNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(value: JsDataInstance) -> JsConstantNodeStub {
        let cloned = value.value.clone();
        JsConstantNodeStub {
            value,
            stub: canvas::NodeStub::Constant(cloned).into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = OkOrCrashNodeStub)]
pub struct JsOkOrCrashNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub message: JsString,
}

#[wasm_bindgen(js_class = OkOrCrash)]
impl JsOkOrCrashNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(msg: JsString) -> JsOkOrCrashNodeStub {
        let rust_msg = String::from(&msg).into();
        JsOkOrCrashNodeStub {
            message: msg,
            stub: canvas::NodeStub::OkOrCrash { msg: rust_msg }.into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = CrashNodeStub)]
pub struct JsCrashNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub message: JsString,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = CrashNode)]
impl JsCrashNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(msg: JsString) -> JsCrashNodeStub {
        let rust_msg = String::from(&msg).into();
        JsCrashNodeStub {
            message: msg,
            stub: canvas::NodeStub::Crash { msg: rust_msg }.into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = UnreachableNodeStub)]
pub struct JsUnreachableNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub message: JsString,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = UnreachableNodeStub)]
impl JsUnreachableNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(msg: JsString) -> JsUnreachableNodeStub {
        let rust_msg = String::from(&msg).into();
        JsUnreachableNodeStub {
            message: msg,
            stub: canvas::NodeStub::Unreachable { msg: rust_msg }.into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = TodoNodeStub)]
pub struct JsTodoNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub message: JsString,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = TodoNodeStub)]
impl JsTodoNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(msg: JsString, inputs: usize) -> JsTodoNodeStub {
        let rust_msg = String::from(&msg).into();
        JsTodoNodeStub {
            message: msg,
            stub: canvas::NodeStub::Todo {
                msg: rust_msg,
                inputs: inputs as canvas::PinOrder,
            }
            .into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = CommentNodeStub)]
pub struct JsCommentNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub message: JsString,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = CommentNodeStub)]
impl JsCommentNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(msg: JsString, inputs: usize) -> JsCommentNodeStub {
        let rust_msg = String::from(&msg).into();
        JsCommentNodeStub {
            message: msg,
            stub: canvas::NodeStub::Comment {
                msg: rust_msg,
                inputs: inputs as canvas::PinOrder,
            }
            .into(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[wasm_bindgen(js_name = NodeStubKind)]
pub enum JsNodeStubKind {
    File,
    SplitBy,
    Input,
    Drop,
    Output,
    Lowercase,
    Uppercase,
    Strip,
    Compare,
    Ordering,
    Regex,
    FindRecord,
    Map,
    List,
    IfElse,
    OkOrErr,
    Validate,
    SelectFirst,
    ExpectOne,
    ExpectSome,
    Match,
    Func,
    ParseDateTime,
    ParseMonetary,
    ParseInt,
    Constant,
    OkOrCrash,
    Crash,
    Unreachable,
    Todo,
    Comment,
}
