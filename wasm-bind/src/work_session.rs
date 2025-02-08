use std::ops::{Deref, DerefMut};
use std::sync::{OnceLock, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use base::canvas::{EdgeNotFoundError, NodeNotFoundError};
use smallvec::SmallVec;
use uuid::Uuid;

use crate::project::Project;
use crate::*;

static WORK_SESSION: OnceLock<RwLock<WorkSession>> = OnceLock::new();

#[macro_export]
macro_rules! wsr {
    () => {
        crate::work_session::work_session()
            .read()
            .expect(crate::WORK_SESSION_POISONED)
    };
}
pub use wsr;

#[macro_export]
macro_rules! wsw {
    () => {
        crate::work_session::work_session()
            .write()
            .expect(crate::WORK_SESSION_POISONED)
    };
}
pub use wsw;

/// Access current work session lock.
pub fn work_session() -> &'static RwLock<WorkSession> {
    WORK_SESSION.get_or_init(|| RwLock::new(WorkSession::new()))
}

/// Work session that contains all the projects and their history loaded into the
/// application. This is a singleton object.
pub struct WorkSession {
    /// Loaded projects.
    projects: SmallVec<[WorkSessionProject; 1]>,

    /// Index of the current project in the `projects` vector.
    current_project_idx: usize,

    /// Function to call to notify when the state of the work session changes.
    on_state_change: Option<js_sys::Function>,
}

unsafe impl Send for WorkSession {}
unsafe impl Sync for WorkSession {}

impl std::default::Default for WorkSession {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkSession {
    pub fn new() -> Self {
        Self {
            projects: SmallVec::new(),
            current_project_idx: 0,
            on_state_change: None,
        }
    }

    /// Convert UUID of the project into the index. Return [None] if the project is
    /// not loaded into the work session.
    fn project_idx(&self, uuid: Uuid) -> Option<usize> {
        self.projects.iter().position(|p| p.project.uuid() == uuid)
    }

    /// Access the project by its UUID. Returns [None] if the project is not loaded
    /// into the work session.
    pub fn project_by_id_mut(&mut self, uuid: Uuid) -> Option<&mut WorkSessionProject> {
        self.projects.iter_mut().find(|p| p.project.uuid() == uuid)
    }

    /// Access the project by its UUID. Returns [None] if the project is not loaded
    /// into the work session.
    pub fn project_by_id(&self, uuid: Uuid) -> Option<&WorkSessionProject> {
        self.projects.iter().find(|p| p.project.uuid() == uuid)
    }

    /// The currently active project in the work session. Returns [None] if there are no projects.
    pub fn current_project(&self) -> Option<&WorkSessionProject> {
        if self.projects.is_empty() {
            None
        } else {
            Some(&self.projects[self.current_project_idx])
        }
    }

    /// The currently active project in the work session. Returns [None] if there are no projects.
    pub fn current_project_mut(&mut self) -> Option<&mut WorkSessionProject> {
        if self.projects.is_empty() {
            None
        } else {
            Some(&mut self.projects[self.current_project_idx])
        }
    }

    /// Switch the current project to another one loaded into the work session.
    pub fn switch_current_project(&mut self, project: Uuid) -> Result<(), ProjectNotFoundError> {
        let idx = self.project_idx(project).ok_or(ProjectNotFoundError)?;
        self.current_project_idx = idx;
        Ok(())
    }
}

#[derive(Debug)]
pub struct ProjectNotFoundError;

/// A project in the work session. This contains the project data and the history stack of changes.
pub struct WorkSessionProject {
    /// The project data.
    project: Project,

    /// History stack of changes made to the project.
    changes: ChangeStack,
}

impl Deref for WorkSessionProject {
    type Target = Project;

    fn deref(&self) -> &Self::Target {
        &self.project
    }
}

impl DerefMut for WorkSessionProject {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.project
    }
}

impl WorkSessionProject {
    /// Undo the last change.
    /// Returns the position of the undone change in the history stack.
    /// If there are no changes to undo, returns `None`.
    pub fn undo(&mut self) -> Option<usize> {
        if self.changes.pos == 0 {
            return None;
        }

        self.changes.pos -= 1;
        let change = self.changes.stack[self.changes.pos].op.clone();
        change.revert(self);
        Some(self.changes.pos)
    }

    /// Redo the last undone change.
    /// Returns the position of the redone change in the history stack.
    /// If there are no changes to redo, returns `None`.
    pub fn redo(&mut self) -> Option<usize> {
        if self.changes.pos == self.changes.stack.len() {
            return None;
        }

        let change = self.changes.stack[self.changes.pos].op.clone();
        change.apply(self);
        self.changes.pos += 1;
        Some(self.changes.pos)
    }

    /// Go to a specific change.
    /// Where 0 is the first change after the initial state.
    /// Returns the position starting from which the changes were undone or redone (position before
    /// the change). If the passed position is out of bounds, this is no-op and returns `None`.
    pub fn goto(&mut self, pos: usize) -> Option<usize> {
        if pos >= self.changes.stack.len() {
            return None;
        }

        if pos < self.changes.pos {
            while self.changes.pos != pos {
                self.undo();
            }
        } else if pos > self.changes.pos {
            while self.changes.pos != pos {
                self.redo();
            }
        }

        Some(self.changes.pos)
    }

    /// Get the change at a specific position in the history stack.
    pub fn change_at(&self, pos: usize) -> Option<&ChangeItem> {
        self.changes.stack.get(pos)
    }

    /// Detect changes in the stack of changes. This is used to detect if the project
    /// was changed since some state, either because of new operations, or because
    /// of undo/redo operations.
    pub fn detect_changed_stack(&self) -> DetectChangedStack {
        DetectChangedStack {
            pos: self.changes.pos,
            time: self
                .changes
                .stack
                .last()
                .map(|c| c.timestamp)
                .unwrap_or(SystemTime::now()),
        }
    }

    /// Record given change operation to the project history.
    fn record(&mut self, op: ChangeOp) {
        self.changes.stack.truncate(self.changes.pos);
        self.changes.stack.push(ChangeItem {
            timestamp: std::time::SystemTime::now(),
            op,
        });
        self.changes.pos += 1;
    }

    pub fn add_node(
        &mut self,
        stub: base::canvas::NodeStub,
        meta: serde_json::Value,
    ) -> base::canvas::Id {
        let id = self
            .project
            .canvas_mut()
            .add_node(stub.clone(), meta.clone());
        self.record(ChangeOp::AddNode {
            stub: Box::new(stub),
            meta,
            id,
        });
        id
    }

    pub fn remove_node(&mut self, id: base::canvas::Id) -> Result<(), NodeNotFoundError> {
        let (node, removed_edges) = self.project.canvas_mut().remove_node(id)?;
        self.record(ChangeOp::RemoveNode {
            removed_edges,
            stub: Box::new(node.stub),
            meta: node.meta,
            id,
        });
        Ok(())
    }

    pub fn add_edge(&mut self, edge: base::canvas::Edge) {
        let op = ChangeOp::AddEdge { edge };
        let outcome_op = op.apply(self);
        self.record(outcome_op);
    }

    pub fn remove_edge(&mut self, edge: base::canvas::Edge) -> Result<(), EdgeNotFoundError> {
        self.project.canvas_mut().remove_edge(edge)?;
        self.record(ChangeOp::RemoveEdge { edge });
        Ok(())
    }

    pub fn patch_node_meta(
        &mut self,
        node_id: base::canvas::Id,
        patch: serde_json::Value,
    ) -> Result<(), NodeNotFoundError> {
        let backup = {
            let node = self
                .project
                .canvas_mut()
                .node_mut(node_id)
                .ok_or(NodeNotFoundError(node_id))?;
            let backup = node.meta.clone();
            json_patch::merge(&mut node.meta, &patch);
            backup
        };
        self.record(ChangeOp::AlterNodeMetadata {
            node_id,
            patch,
            backup,
        });
        Ok(())
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
        let ws = wsr!();
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
        let mut ws = wsw!();
        ws.switch_current_project(project.uuid)
            .map_err(|_| JsError::new("Project not found"))
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
            return Err(JsError::new(
                "Fake server mode throws an error on save attempt",
            ));
        }

        todo!()
    }

    /// Set a callback to be called when the state of the work session changes.
    /// This includes interruptions in network connection, background validation run,
    /// background downloads, etc.
    #[wasm_bindgen(setter, js_name = onStateChange)]
    pub fn set_on_state_change(&mut self, f: js_sys::Function) {
        let mut ws = wsw!();
        ws.on_state_change = Some(f);
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
        let mut ws = wsw!();
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
        let mut ws = wsw!();
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
        let mut ws = wsw!();
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
    /// It is used internally to verify the validity of the handle.
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
        let ws = wsr!();
        if let Some(project) = ws.project_by_id(self.project_uuid) {
            if let Some(change) = project.change_at(self.position) {
                change.micros_since_unix() as u64 == self.timestamp
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Get the operation that was performed.
    /// This information is necessary to reflect the change in the UI.
    /// If the handle is invalid, this will return an error.
    pub fn operation(&self) -> Result<JsChangeOp, InvalidHandleError> {
        if !self.is_valid() {
            return Err(InvalidHandleError);
        }

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
    /// Stack of changes made to the project.
    stack: Vec<ChangeItem>,

    /// Current position in the history stack, that is reflected on the canvas
    /// of the project and, consequently, in the UI.
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

/// Detect changes in the stack of changes. This is used to detect if the project
/// was changed since some state.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DetectChangedStack {
    /// Position in the history stack.
    pos: usize,

    /// Timestamp of the last change.
    time: std::time::SystemTime,
}

pub struct ChangeItem {
    timestamp: std::time::SystemTime,
    op: ChangeOp,
}

impl ChangeItem {
    /// Get the timestamp of the change in microseconds since UNIX epoch.
    pub fn micros_since_unix(&self) -> u128 {
        self.timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros()
    }
}

/// Change operation that can be performed on the project. This
/// information is enough to reflect the change in the UI.
/// This allows as well to revert the existing change.
#[derive(Debug, Clone)]
pub enum ChangeOp {
    /// Add a node to the canvas.
    AddNode {
        stub: Box<base::canvas::NodeStub>,
        meta: serde_json::Value,
        id: base::canvas::Id,
    },

    /// Remove a node from the canvas.
    RemoveNode {
        removed_edges: Vec<base::canvas::Edge>,
        stub: Box<base::canvas::NodeStub>,
        meta: serde_json::Value,
        id: base::canvas::Id,
    },

    /// Add an edge to the canvas.
    AddEdge { edge: base::canvas::Edge },

    /// Remove an edge from the canvas.
    RemoveEdge { edge: base::canvas::Edge },

    /// Alter metadata of a node.
    AlterNodeMetadata {
        /// Node in which the metadata was changed.
        node_id: base::canvas::Id,

        /// Values to (re)set in the metadata.
        ///
        /// For edits, the value is the new value intead of the existing one.
        /// All null values in a map are removed from the metadata.
        patch: serde_json::Value,

        /// Pre-patch values to revert the metadata back to the original state.
        backup: serde_json::Value,
    },
}

impl ChangeOp {
    /// Apply the change operation to the project, returning the resulting actual
    /// operation. Note that the operation can be different (with different params)
    /// than the original one, e.g. ID of the new node can change.
    ///
    /// # Panic
    /// This method panics if the operation cannot be applied, which cannot
    /// happen if the operation was recorded correctly and when all changes are
    /// tracked correctly.
    fn apply(self, project: &mut WorkSessionProject) -> ChangeOp {
        use ChangeOp::*;

        const EXPECT_FOUND_NODE: &str = "node not found, though operation was recorded";
        const EXPECT_FOUND_EDGE: &str = "edge not found, though operation was recorded";

        match self {
            AddNode { stub, id: _, meta } => {
                let new_id = project
                    .project
                    .canvas_mut()
                    .add_node(stub.as_ref().clone(), meta.clone());
                AddNode {
                    stub,
                    meta,
                    id: new_id,
                }
            }
            RemoveNode {
                removed_edges: _,
                meta: _,
                mut stub,
                id,
            } => {
                let (node, removed_edges) = project
                    .project
                    .canvas_mut()
                    .remove_node(id)
                    .expect(EXPECT_FOUND_NODE);

                // Reuse existing "Box", but we don't care about that dummy input stub.
                *stub = node.stub;

                RemoveNode {
                    removed_edges,
                    stub,
                    meta: node.meta,
                    id,
                }
            }
            AddEdge { edge } => {
                project
                    .project
                    .canvas_mut()
                    .add_edge(edge)
                    .expect(EXPECT_FOUND_EDGE);
                AddEdge { edge }
            }
            RemoveEdge { edge } => {
                project
                    .project
                    .canvas_mut()
                    .remove_edge(edge)
                    .expect(EXPECT_FOUND_EDGE);
                RemoveEdge { edge }
            }
            AlterNodeMetadata {
                node_id,
                patch,
                backup: _,
            } => {
                let meta = &mut project
                    .project
                    .canvas_mut()
                    .node_mut(node_id)
                    .expect(EXPECT_FOUND_NODE)
                    .meta;
                let backup = meta.clone();

                json_patch::merge(meta, &patch);

                AlterNodeMetadata {
                    node_id,
                    patch,
                    backup,
                }
            }
        }
    }

    /// Revert the change operation from the project.
    /// This is the opposite of [ChangeOp::apply].
    fn revert(self, project: &mut WorkSessionProject) {
        use ChangeOp::*;

        const EXPECT_FOUND_NODE: &str = "node not found, though operation was recorded";
        const EXPECT_FOUND_EDGE: &str = "edge not found, though operation was recorded";

        match self {
            AddNode { id, .. } => {
                project
                    .project
                    .canvas_mut()
                    .remove_node(id)
                    .expect(EXPECT_FOUND_NODE);
            }
            RemoveNode {
                removed_edges,
                stub,
                meta,
                id: _,
            } => {
                project.project.canvas_mut().add_node(*stub, meta);

                for edge in removed_edges {
                    project
                        .project
                        .canvas_mut()
                        .add_edge(edge)
                        .expect(EXPECT_FOUND_EDGE);
                }
            }
            AddEdge { edge } => {
                project
                    .project
                    .canvas_mut()
                    .remove_edge(edge)
                    .expect(EXPECT_FOUND_EDGE);
            }
            RemoveEdge { edge } => {
                project
                    .project
                    .canvas_mut()
                    .add_edge(edge)
                    .expect(EXPECT_FOUND_EDGE);
            }
            AlterNodeMetadata {
                node_id,
                patch: _,
                backup,
            } => {
                let meta = &mut project
                    .project
                    .canvas_mut()
                    .node_mut(node_id)
                    .expect(EXPECT_FOUND_NODE)
                    .meta;
                *meta = backup;
            }
        }
    }
}
