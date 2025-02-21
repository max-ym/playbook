//! This project is designed to interoperate with frontend JS side to provide
//! a bridge between Rust implementation of validation logic and the frontend
//! implementation of the UI.
//! 
//! # Naming
//! All JS-related structures have "Js" prefix to avoid name conflicts with
//! Rust structures. For example, [JsProject](project::JsProject) is a JS-side
//! handle for a [Project](project::Project) structure. The JS-side handle
//! is visible from the JS side as `Project` though.
//! 
//! # Fake Server
//! The fake server mode will be applied if WASM package is compiled with corresponding
//! feature enabled.
//! 
//! When the project is compiled to WASM, it will provide a fake server implementation
//! that simulates some calls to the server with predefined data set. For functions that
//! support fake server mode, the documentation mentions of how it is simulated
//! and which arguments are allowed to be passed. Note that those functions may
//! not support all features of actual server implementation, but are designed
//! to aid testing and development.
//! 
//! ## Basic Use Flow
//! 1. List available project - not currently implemented!
//! 2. Load project by zero UUID - [JsProject::load](project::JsProject::load).
//!    [Work Session](work_session::JsWorkSession) will then have a project available to work with.
//!    Since there is only one project loaded, it will be automatically selected as
//!    the current project.
//! 3. Use [Canvas](project::JsCanvas) for reading and editing nodes and edges.
//! 4. Use [Work Session History](work_session::JsHistory) to undo and redo changes of the
//!    current project. Currently, it does not support actually providing meaningful
//!    information about change itself, but undo/redo operations would perform the changes
//!    over the canvas. Full reload of the canvas would allow to render the changes for now.

use js_sys::JsString;
use thiserror::Error;
use wasm_bindgen::prelude::*;

pub mod work_session;

pub mod project;

pub mod vcs;

/// To use with `expect` method on operations on `RwLock` of `WorkSession`.
/// This should never happen, but if it does, it means that the WASM code panicked
/// and the session is in an invalid state.
const WORK_SESSION_POISONED: &str =
    "work session is poisoned (unexpected crash happened in WASM), reload required";

#[derive(Debug, Error)]
#[wasm_bindgen]
#[error("handle is invalid and can't access the original object")]
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
