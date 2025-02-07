use std::sync::{OnceLock, RwLock};
use std::time::UNIX_EPOCH;

use smallvec::SmallVec;
use uuid::Uuid;

use crate::project::Project;
use crate::*;

static WORK_SESSION: OnceLock<RwLock<WorkSession>> = OnceLock::new();

pub fn work_session() -> &'static RwLock<WorkSession> {
    WORK_SESSION.get_or_init(|| RwLock::new(WorkSession::new()))
}

pub struct WorkSession {
    /// Loaded projects.
    projects: SmallVec<[WorkSessionProject; 1]>,

    /// Index of the current project in the `projects` vector.
    current_project_idx: usize,

    on_state_change: Option<js_sys::Function>,
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

    fn project_idx(&self, uuid: Uuid) -> Option<usize> {
        self.projects.iter().position(|p| p.project.uuid() == uuid)
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

    pub fn current_project(&self) -> Option<&WorkSessionProject> {
        if self.projects.is_empty() {
            None
        } else {
            Some(&self.projects[self.current_project_idx])
        }
    }

    pub fn current_project_mut(&mut self) -> Option<&mut WorkSessionProject> {
        if self.projects.is_empty() {
            None
        } else {
            Some(&mut self.projects[self.current_project_idx])
        }
    }
}

pub struct WorkSessionProject {
    project: Project,
    changes: ChangeStack,
}

impl WorkSessionProject {
    /// Undo the last change.
    /// Returns the position of the undone change in the history stack.
    /// If there are no changes to undo, returns `None`.
    pub fn undo(&mut self) -> Option<usize> {
        todo!()
    }

    /// Redo the last undone change.
    /// Returns the position of the redone change in the history stack.
    /// If there are no changes to redo, returns `None`.
    pub fn redo(&mut self) -> Option<usize> {
        todo!()
    }

    /// Go to a specific change.
    /// Where 0 is the first change after the initial state.
    /// Returns the position starting from which the changes were undone or redone (position before
    /// the change). If the passed position is out of bounds, this is no-op and returns `None`.
    pub fn goto(&mut self, pos: usize) -> Option<usize> {
        todo!()
    }

    pub fn change_at(&self, pos: usize) -> Option<&ChangeItem> {
        self.changes.stack.get(pos)
    }

    pub fn uuid(&self) -> Uuid {
        self.project.uuid()
    }
}

#[derive(Debug, Copy, Clone)]
#[wasm_bindgen(js_name = WorkSession)]
pub struct JsWorkSession;

#[wasm_bindgen(js_class = WorkSession)]
impl JsWorkSession {
    /// Get the current work session.
    /// This throws an exception `WorkSessionUninitError` if the work session is not initialized
    /// or is in an invalid state.
    pub fn get() -> Result<Self, JsWorkSessionUninitError> {
        // TODO
        Ok(Self)
    }

    /// Get the current project in the work session. Returns `undefined` if there are no projects.
    #[wasm_bindgen(getter, js_name = currentProject)]
    pub fn current_project(&self) -> Option<project::JsProject> {
        let ws = work_session().read().expect(WORK_SESSION_POISONED);
        let project = ws.current_project()?;
        Some(project::JsProject {
            uuid: project.uuid(),
        })
    }

    /// Select another project to be "current", e.g. the one that is currently being shown
    /// on the UI. This in turn will affect on what global work session history shows.
    /// Switch of the current project will also switch all related sub-resources like
    /// project history (undo/redo stack), etc. Old changes are still preserved and can be
    /// recovered by switching back to the previous project.
    #[wasm_bindgen(js_name = switchCurrentProject)]
    pub fn switch_current_project(&self, project: project::JsProject) -> Result<(), JsError> {
        let mut ws = work_session().write().expect(WORK_SESSION_POISONED);
        let idx = ws.project_idx(project.uuid).ok_or(JsError::new("Project not found"))?;
        ws.current_project_idx = idx;
        Ok(())
    }

    /// Check whether all current changes (as a draft) are synchronized with the server.
    /// This is useful when the user has sudden power loss, network disconnect, browser
    /// crash, for the changes to remain recoverable.
    /// True means all data is saved on the server, false means there are unsaved changes.
    /// 
    /// # Fake Server
    /// In the fake server mode, this always returns false. This aligns with the
    /// `saveToServer` always throwing an error in the fake server mode.
    #[wasm_bindgen(getter, js_name = isServerSaved)]
    pub fn is_server_saved() -> bool {
        if cfg!(feature = "fake_server") {
            return false;
        }

        todo!()
    }

    /// Force save all current changes to the server.
    /// This is useful when the user wants to make sure that all changes are saved e.g. when
    /// closing the browser tab.
    /// 
    /// # Errors
    /// This will throw an error if the server is not reachable,
    /// or if the user is not authenticated.
    /// 
    /// # Fake Server
    /// In the fake server mode, this will throw an error.
    #[wasm_bindgen(js_name = saveToServer)]
    pub async fn save_to_server(&self) -> Result<(), JsError> {
        if cfg!(feature = "fake_server") {
            return Err(JsError::new("Fake server mode throws an error on save attempt"));
        }

        todo!()
    }

    /// Set a callback to be called when the state of the work session changes.
    /// This includes interruptions in network connection, background validation run,
    /// background downloads, etc.
    #[wasm_bindgen(setter, js_name = onStateChange)]
    pub fn set_on_state_change(&mut self, f: js_sys::Function) {
        todo!()
    }
}

/// Global history stack of changes made to the current project.
/// This allows to undo and redo changes of the project on the screen.
/// This is a singleton object.
#[derive(Debug, Copy, Clone)]
#[wasm_bindgen(js_name = History)]
pub struct JsHistory;

#[wasm_bindgen(js_class = History)]
impl JsHistory {
    /// Undo the last change in the current project.
    /// Returns the undone change in the history stack.
    /// If there are no changes to undo, returns `undefined`.
    pub fn undo() -> Option<JsChangeItem> {
        let mut ws = work_session().write().expect(WORK_SESSION_POISONED);
        let project = ws.current_project_mut()?;
        let position = project.undo()?;
        project.change_at(position).map(|change| JsChangeItem {
            timestamp: change.micros_since_unix() as u64,
            position,
            project_uuid: project.uuid(),
        })
    }

    /// Redo the last undone change in the current project.
    /// Returns the redone change in the history stack.
    /// If there are no changes to redo, returns `undefined`.
    pub fn redo() -> Option<JsChangeItem> {
        let mut ws = work_session().write().expect(WORK_SESSION_POISONED);
        let project = ws.current_project_mut()?;
        let position = project.redo()?;
        project.change_at(position).map(|change| JsChangeItem {
            timestamp: change.micros_since_unix() as u64,
            position,
            project_uuid: project.uuid(),
        })
    }

    /// Go to a specific change in the current project. 
    /// Where 0 is the first change after the initial state.
    /// Returns the position starting from which the changes were undone or redone (position before
    /// the change). If the passed position is out of bounds, this is no-op and returns `undefined`.
    pub fn goto(pos: usize) -> Option<usize> {
        let mut ws = work_session().write().expect(WORK_SESSION_POISONED);
        let project = ws.current_project_mut()?;
        project.goto(pos)
    }

    /// Get the change at a specific position in the history stack.
    /// If the position is out of bounds, returns `undefined`.
    pub fn change_at(pos: usize) -> Option<JsChangeItem> {
        let ws = work_session().read().expect(WORK_SESSION_POISONED);
        let project = ws.current_project()?;
        project.change_at(pos).map(|change| JsChangeItem {
            timestamp: change.micros_since_unix() as u64,
            position: pos,
            project_uuid: project.uuid(),
        })
    }
}

/// A change in the project. This defines the operation that was performed, and that
/// can be undone or redone.
#[derive(Debug)]
#[wasm_bindgen(js_name = ChangeItem)]
pub struct JsChangeItem {
    /// Timestamp of the change.
    #[wasm_bindgen(readonly)]
    pub timestamp: u64,

    /// Position of the change in the history stack.
    /// This together with the timestamp can be used to uniquely identify a change.
    ///
    /// During execution, if it so happens that this position points to a different
    /// change (identifiable by the timestamp), this means that the history stack
    /// was modified in the meantime. Then this handle is invalid and further operations
    /// on it will result in an error.
    position: usize,

    project_uuid: Uuid,
}

#[wasm_bindgen(js_class = ChangeItem)]
impl JsChangeItem {
    /// Check if the change handle is still valid.
    ///
    /// During execution, the history stack can get modified. Then this handle can get invalid if
    /// the corresponding change is overwritten, so further operations
    /// on it will result in an error.
    #[wasm_bindgen(getter, js_name = isValid)]
    pub fn is_valid(&self) -> bool {
        todo!()
    }

    /// Get the operation that was performed.
    /// This information is necessary to reflect the change in the UI.
    /// If the handle is invalid, this will return an error.
    pub fn operation(&self) -> Result<JsChangeOp, InvalidHandleError> {
        todo!()
    }
}

/// Change operation that was performed on the project with related data. This
/// information is necessary to reflect the change in the UI.
#[derive(Debug)]
#[wasm_bindgen(js_name = ChangeOp)]
pub struct JsChangeOp {
    // TODO
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
    timestamp: std::time::SystemTime,
    op: ChangeOp,
}

impl ChangeItem {
    /// Get the timestamp of the change in microseconds since UNIX epoch.
    pub fn micros_since_unix(&self) -> u128 {
        self.timestamp.duration_since(UNIX_EPOCH).unwrap_or_default().as_micros()
    }
}

pub enum ChangeOp {}
