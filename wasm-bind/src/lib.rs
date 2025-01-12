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

    fn new_js(js: JsString) -> Self {
        Self {
            rust: CompactString::new(""),
            js,
            sync: InterStrSync::Js,
        }
    }

    pub fn make_js(&mut self) -> &JsString {
        match self.sync {
            InterStrSync::Rust => {
                self.js = JsString::from(self.rust.as_str());
            }
            InterStrSync::Both => {}
            InterStrSync::Js => {}
        }
        self.sync = InterStrSync::Both;
        &self.js
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

fn vec_i8_into_u8(v: Vec<i8>) -> Vec<u8> {
    // Ideally we'd use Vec::into_raw_parts, but it's unstable,
    // so we have to do it manually:

    // First, make sure v's destructor doesn't free the data
    // it thinks it owns when it goes out of scope.
    let mut v = std::mem::ManuallyDrop::new(v);

    // then, pick apart the existing Vec
    let p = v.as_mut_ptr();
    let len = v.len();
    let cap = v.capacity();
    
    // finally, adopt the data into a new Vec
    unsafe { Vec::from_raw_parts(p as *mut u8, len, cap) }
}
