use std::sync::{OnceLock, RwLock};

use smallvec::SmallVec;
use uuid::Uuid;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::project::Project;

static WORK_SESSION: OnceLock<RwLock<WorkSession>> = OnceLock::new();

pub fn work_session() -> &'static RwLock<WorkSession> {
    WORK_SESSION.get_or_init(|| RwLock::new(WorkSession::new()));
    WORK_SESSION.get().unwrap()
}

pub struct WorkSession {
    /// Loaded projects.
    projects: SmallVec<[WorkSessionProject; 1]>,

    /// Index of the current project in the `projects` vector.
    current_project_idx: usize,

    on_state_change: Option<js_sys::Function>,
}

struct WorkSessionProject {
    project: Project,
    changes: ChangeStack,
}

unsafe impl Send for WorkSession {}
unsafe impl Sync for WorkSession {}

impl WorkSession {
    pub fn new() -> Self {
        Self {
            projects: SmallVec::new(),
            current_project_idx: 0,
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

    pub fn project_by_id_mut(&mut self, uuid: Uuid) -> Option<&mut crate::project::Project> {
        self.projects
            .iter_mut()
            .find(|p| p.project.uuid() == uuid)
            .map(|p| &mut p.project)
    }

    pub fn project_by_id(&self, uuid: Uuid) -> Option<&crate::project::Project> {
        self.projects
            .iter()
            .find(|p| p.project.uuid() == uuid)
            .map(|p| &p.project)
    }
}

#[derive(Debug, Copy, Clone)]
#[wasm_bindgen(js_name = WorkSession)]
pub struct JsWorkSession;

#[wasm_bindgen]
impl JsWorkSession {
    pub fn get() -> Result<Self, JsWorkSessionUninitError> {
        todo!()
    }
}

#[derive(Debug, Copy, Clone)]
#[wasm_bindgen(js_name = History)]
pub struct JsHistory;

#[wasm_bindgen]
impl JsHistory {
    /// Undo the last change.
    pub fn undo(&self) -> bool {
        work_session().write().unwrap().undo()
    }

    /// Redo the last undone change.
    pub fn redo(&self) -> bool {
        work_session().write().unwrap().redo()
    }

    /// Go to a specific change. Where 0 is the first change after the initial state.
    pub fn goto(&self, pos: usize) -> bool {
        work_session().write().unwrap().goto(pos)
    }
}

/// Error when trying to access an uninitialized or incorrectly initialized work session.
#[wasm_bindgen(js_name = WorkSessionUninitError)]
pub struct JsWorkSessionUninitError {
    /// Work session to use with possibly invalid state.
    #[wasm_bindgen(js_name = workSession)]
    pub work_session: JsWorkSession,
    // TODO more
}

/// Stack of changes made to a project. This allows to undo and redo changes.
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
