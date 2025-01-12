use std::sync::OnceLock;

use serde_json::Value as JsonValue;
use tokio::sync::RwLock;
use wasm_bindgen::prelude::wasm_bindgen;

static WORK_SESSION: OnceLock<RwLock<WorkSession>> = OnceLock::new();

fn work_session() -> &'static RwLock<WorkSession> {
    unsafe {
        WORK_SESSION.get_or_init(|| RwLock::new(WorkSession::new()));
        WORK_SESSION.get().unwrap_unchecked()
    }
}

pub struct WorkSession {
    canvas: base::canvas::Canvas<JsonValue>,
    changes: ChangeStack,
    on_state_change: Option<js_sys::Function>,
}

unsafe impl Send for WorkSession {}
unsafe impl Sync for WorkSession {}

impl WorkSession {
    pub fn new() -> Self {
        Self {
            canvas: base::canvas::Canvas::new(),
            changes: ChangeStack::new(),
            on_state_change: None,
        }
    }

    pub fn undo(&mut self) -> bool {
        todo!()
    }

    pub fn redo(&mut self) -> bool {
        todo!()
    }

    pub fn goto(&self, pos: usize) -> bool {
        todo!()
    }
}

#[derive(Debug, Copy, Clone)]
#[wasm_bindgen(js_name = WorkSession)]
pub struct JsWorkSession;

#[wasm_bindgen]
impl JsWorkSession {
    pub fn get() -> Result<Self, Self> {
        todo!()
    }

    pub fn history(&self) -> JsHistory {
        JsHistory
    }

    pub fn canvas(&self) -> JsCanvas {
        todo!()
    }
}

#[derive(Debug, Copy, Clone)]
#[wasm_bindgen(js_name = Canvas)]
pub struct JsCanvas;

#[wasm_bindgen]
impl JsCanvas {}

#[derive(Debug, Copy, Clone)]
#[wasm_bindgen(js_name = History)]
pub struct JsHistory;

#[wasm_bindgen]
impl JsHistory {
    pub async fn undo(&self) -> bool {
        work_session().write().await.undo()
    }

    pub async fn redo(&self) -> bool {
        work_session().write().await.redo()
    }

    pub async fn goto(&self, pos: usize) -> bool {
        work_session().write().await.goto(pos)
    }
}

#[wasm_bindgen(js_name = WorkSessionUninitError)]
pub struct JsWorkSessionUninitError {
    #[wasm_bindgen(js_name = workSession)]
    pub work_session: JsWorkSession,
    // TODO more
}

pub struct ChangeStack {
    stack: Vec<ChangeItem>,
    pos: usize,
}

impl ChangeStack {
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            pos: 0,
        }
    }
}

pub struct ChangeItem {
    timestamp: std::time::Instant,
    op: ChangeOp,
}

pub enum ChangeOp {}
