use compact_str::CompactString;
use js_sys::JsString;
use wasm_bindgen::{prelude::*, JsCast, JsValue};

pub mod work_session;

pub mod project;

// #[derive(Debug)]
// pub enum JsResult<T: Into<JsValue>, E: Into<JsValue>> {
//     Ok(T),
//     Err(E),
// }

// #[wasm_bindgen(js_name = Result)]
// pub struct JsResultExported {
//     /// This will be undefined if the result is an ok.
//     err: JsValue,

//     /// This will be undefined if the result is error.
//     ok: JsValue,
// }

// impl<T: Into<JsValue>, E: Into<JsValue>> From<JsResult<T, E>> for JsResultExported {
//     fn from(result: JsResult<T, E>) -> Self {
//         match result {
//             JsResult::Ok(ok) => Self {
//                 err: JsValue::undefined(),
//                 ok: ok.into(),
//             },
//             JsResult::Err(err) => Self {
//                 err: err.into(),
//                 ok: JsValue::undefined(),
//             },
//         }
//     }
// }

/// String representation that can be used in both Rust and JS.
/// It prevents unnecessary conversions between Rust and JS strings, as those
/// are done lazily here.
#[derive(Debug, Clone)]
pub struct InterString {
    rust: CompactString,
    js: JsString,

    /// Whether the string is correct in Rust or JS or both.
    sync: InterStrSync,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum InterStrSync {
    Rust,
    Js,
    Both,
}

impl InterString {
    fn new() -> Self {
        Self {
            rust: CompactString::new(""),
            js: JsString::from(""),
            sync: InterStrSync::Both,
        }
    }

    fn js(&self) -> Option<JsString> {
        match self.sync {
            InterStrSync::Js | InterStrSync::Both => Some(self.js.clone()),
            InterStrSync::Rust => None,
        }
    }

    fn to_js(&self) -> JsString {
        if let Some(js) = self.js() {
            js
        } else {
            JsString::from(self.rust.as_str())
        }
    }

    fn new_js(js: JsString) -> Self {
        Self {
            rust: CompactString::new(""),
            js,
            sync: InterStrSync::Js,
        }
    }

    pub fn make_js(&mut self) -> JsString {
        match self.sync {
            InterStrSync::Rust => {
                self.js = JsString::from(self.rust.as_str());
            }
            InterStrSync::Both => {}
            InterStrSync::Js => {}
        }
        self.sync = InterStrSync::Both;
        self.js.clone()
    }

    pub fn make_rust(&mut self) -> &CompactString {
        match self.sync {
            InterStrSync::Js => {
                let len = self.js.length() as usize;
                let mut s = CompactString::with_capacity(len);
                for ch in char::decode_utf16(self.js.iter()) {
                    match ch {
                        Ok(ch) => s.push(ch),
                        Err(_) => s.push(std::char::REPLACEMENT_CHARACTER),
                    }
                }

                self.rust = s;
            }
            InterStrSync::Both => {}
            InterStrSync::Rust => {}
        }
        self.sync = InterStrSync::Both;
        &self.rust
    }

    pub fn set_js(&mut self, js: JsString) {
        self.js = js;
        self.sync = InterStrSync::Js;
    }

    pub fn make_mut_rust(&mut self) -> &mut CompactString {
        self.make_rust();
        self.sync = InterStrSync::Rust;
        &mut self.rust
    }
}

impl Default for InterString {
    fn default() -> Self {
        Self::new()
    }
}

impl From<JsString> for InterString {
    fn from(js: JsString) -> Self {
        Self::new_js(js)
    }
}

impl From<String> for InterString {
    fn from(rust: String) -> Self {
        rust.as_str().into()
    }
}

impl From<&str> for InterString {
    fn from(rust: &str) -> Self {
        Self {
            rust: CompactString::new(rust),
            js: JsString::from(rust),
            sync: InterStrSync::Both,
        }
    }
}

impl From<CompactString> for InterString {
    fn from(rust: CompactString) -> Self {
        let js = JsString::from(rust.as_str());
        Self {
            rust,
            js,
            sync: InterStrSync::Both,
        }
    }
}

/// Error for failure to reconstruct a type from a JS value.
pub enum TypeReconstructionError {
    MissingField(&'static str),
    InvalidType(&'static str),
    UnexpectedVariant(&'static str),
}

impl TypeReconstructionError {
    pub fn ensure_string(js: JsValue) -> Result<JsString, TypeReconstructionError> {
        js.dyn_into().map_err(|_| TypeReconstructionError::InvalidType("String"))
    }
}

trait MyUuid {
    fn into_js_array(self) -> js_sys::Uint8Array;
}

impl MyUuid for uuid::Uuid {
    fn into_js_array(self) -> js_sys::Uint8Array {
        let bytes = self.as_bytes();
        let arr = js_sys::Uint8Array::new_with_length(bytes.len() as u32);
        for (i, byte) in bytes.iter().enumerate() {
            arr.set_index(i as u32, *byte);
        }
        arr
    }
}

#[wasm_bindgen]
#[derive(Debug)]
pub struct PermissionError;

// JS standard File type.
#[wasm_bindgen]
extern "C" {
    pub type File;

    #[wasm_bindgen(method, getter)]
    fn name(this: &File) -> JsString;

    #[wasm_bindgen(method, getter)]
    fn bytes(this: &File) -> js_sys::Uint8Array;
}
