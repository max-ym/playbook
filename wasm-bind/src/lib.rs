use js_sys::JsString;
use wasm_bindgen::prelude::*;

pub mod work_session;

pub mod project;

pub mod vcs;

/// To use with `expect` method on operations on `RwLock` of `WorkSession`.
/// This should never happen, but if it does, it means that the WASM code panicked
/// and the session is in an invalid state.
const WORK_SESSION_POISONED: &str =
    "work session is poisoned (unexpected crash happened in WASM), reload required";

#[wasm_bindgen]
pub struct InvalidHandleError;

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
