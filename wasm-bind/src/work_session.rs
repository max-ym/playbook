use std::ops::{Deref, DerefMut};
use std::sync::{OnceLock, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use base::canvas;
use canvas::{EdgeNotFoundError, NodeNotFoundError};
use log::{debug, error, trace, warn};
use smallvec::SmallVec;
use uuid::Uuid;

use crate::project::{Metadata, Project};
use crate::*;

/// Change operations that can be performed on the project.
mod change_op;
pub use change_op::{ChangeOp, ChangeOpStub};

static WORK_SESSION: OnceLock<RwLock<WorkSession>> = OnceLock::new();

#[macro_export]
macro_rules! wsr {
    () => {{
        log::trace!("read-lock work session singleton");
        crate::work_session::work_session()
            .read()
            .expect(crate::WORK_SESSION_POISONED)
    }};
}
pub use wsr;

#[macro_export]
macro_rules! wsw {
    () => {{
        log::trace!("write-lock work session singleton");
        crate::work_session::work_session()
            .write()
            .expect(crate::WORK_SESSION_POISONED)
    }};
}
pub use wsw;

/// Access current work session lock.
///
/// This also effectively initializes this WASM package. Currently, it sets up the logger.
pub fn work_session() -> &'static RwLock<WorkSession> {
    WORK_SESSION.get_or_init(|| {
        let _ = console_log::init_with_level(
            log::STATIC_MAX_LEVEL.to_level().unwrap_or(log::Level::Info),
        );
        RwLock::new(WorkSession::new())
    })
}

/// Work session that contains all the projects and their history loaded into the
/// application. This is a singleton object.
pub struct WorkSession {
    /// Loaded projects.
    projects: SmallVec<[WorkSessionProject; 1]>,

    /// Index of the current project in the `projects` vector.
    current_project_idx: usize,
}

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

    /// Add new project to the work session.
    pub fn add_project(&mut self, project: Project) -> Result<(), ProjectExistsError> {
        let uuid = project.uuid();
        if self.project_by_id(uuid).is_some() {
            return Err(ProjectExistsError(uuid));
        }

        self.projects.push(WorkSessionProject {
            project,
            changes: ChangeStack::new(),
        });

        Ok(())
    }
}

/// Error when trying to access a project that is not loaded into the work session.
#[derive(Debug, Error)]
#[error("project not found in the work session")]
pub struct ProjectNotFoundError;

/// Error when trying to add a project that already exists in the work session.
#[derive(Debug, Error)]
#[error("project already exists in the work session by UUID {0}")]
pub struct ProjectExistsError(pub Uuid);

/// A project in the work session. This contains the project data and the history stack of changes.
pub struct WorkSessionProject {
    /// The project data.
    project: Project,

    /// History stack of changes made to the project.
    changes: ChangeStack,
}

// Safety: WorkSessionProject should not shared between threads.
// We have issues here because JavaScript values are internally just
// pointers to the actual data, and we can't guarantee that the data is valid and
// is not shared between threads. Ideally, all calls into this WASM module should be always in the
// same worker thread, and the same object should never be shared between different workers.
// This isolation provides safety against possible issues, which Rust warns us about.
// Since Rust cannot prove this isolation, we have to use `unsafe impl` manually here to lift
// the restriction, promissing to behave...
unsafe impl Send for WorkSessionProject {}
unsafe impl Sync for WorkSessionProject {}

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
            debug!("no changes to undo");
            return None;
        }
        trace!("undoing change");

        todo!()
    }

    /// Redo the last undone change.
    /// Returns the position of the redone change in the history stack.
    /// If there are no changes to redo, returns `None`.
    pub fn redo(&mut self) -> Option<usize> {
        if self.changes.pos >= self.changes.stack.len() {
            debug!("no changes to redo");
            return None;
        }
        trace!("redoing change");

        todo!()
    }

    /// Go to a specific change.
    /// Where 0 is the first change after the initial state.
    /// Returns the position starting from which the changes were undone or redone (position before
    /// the change). If the passed position is out of bounds, this is no-op and returns `None`.
    pub fn goto(&mut self, pos: usize) -> Option<usize> {
        if pos >= self.changes.stack.len() {
            debug!(
                "change stack position out of bounds ({pos} >= {})",
                self.changes.stack.len()
            );
            return None;
        }

        if pos < self.changes.pos {
            trace!("going back to change at position {pos}");
            while self.changes.pos != pos {
                self.undo();
            }
        } else if pos > self.changes.pos {
            trace!("going forward to change at position {pos}");
            while self.changes.pos != pos {
                self.redo();
            }
        } else {
            assert_eq!(
                self.changes.pos, pos,
                "above condition should be exhaustive, hence this should be the only correct state here",
            );
            debug!("`goto` op on history stack found itself already at position {pos}");
        }

        Some(self.changes.pos)
    }

    /// Get the change at a specific position in the history stack.
    pub fn change_at(&self, pos: usize) -> Option<&ChangeItem> {
        self.changes.stack.get(pos)
    }

    /// See [WorkSessionProject::change_at].
    pub fn change_at_mut(&mut self, pos: usize) -> Option<&mut ChangeItem> {
        self.changes.stack.get_mut(pos)
    }

    /// Get the change in the history stack to which the project is currently pointing.
    /// If pointer was not moved by undoing, this is the last change made.
    pub fn current_change(&self) -> Option<&ChangeItem> {
        self.change_at(self.changes.pos)
    }

    /// See [WorkSessionProject::current_change].
    pub fn current_change_mut(&mut self) -> Option<&mut ChangeItem> {
        self.change_at_mut(self.changes.pos)
    }

    /// Checkout changes in the stack of changes. This is used to detect if the project
    /// was changed since some state, either because of new operations, or because
    /// of undo/redo operations.
    pub fn stack_checkout(&self) -> CheckoutChangedStack {
        CheckoutChangedStack {
            pos: self.changes.pos,
            time: self
                .changes
                .stack
                .last()
                .map(|c| c.timestamp)
                .unwrap_or(SystemTime::now()),
        }
    }

    pub fn add_node(&mut self, stub: canvas::NodeStub) -> canvas::Id {
        trace!("add node to the project: {stub:#?}");

        let op = ChangeOp::add_node(self, stub);
        let id = op.id;

        trace!("added node with ID {id}");
        self.changes.push_op(op);
        id
    }

    pub fn remove_node(&mut self, id: canvas::Id) -> Result<(), NodeNotFoundError> {
        trace!("remove node {id} from the project");
        let op = ChangeOp::remove_node(self, id)?;
        self.changes.push_op(op);
        Ok(())
    }

    pub fn add_edge(&mut self, edge: canvas::Edge) -> Result<(), NodeNotFoundError> {
        trace!("add edge {edge} to the project");
        let op = ChangeOp::add_edge(self, edge)?;
        self.changes.push_op(op);
        Ok(())
    }

    pub fn remove_edge(&mut self, edge: canvas::Edge) -> Result<(), EdgeNotFoundError> {
        trace!("remove edge {edge} from the project");
        let op = ChangeOp::remove_edge(self, edge)?;
        self.changes.push_op(op);
        Ok(())
    }

    /// Alter metadata of a node.
    ///
    /// # Flatten
    /// If `flatten` is true and the last operation was also altering the same metadata key
    /// of the same node,
    /// the new value will be merged with the previous one into a single operation.
    /// It has no effect if the previous operation was on the different key.
    ///
    /// Intention here is to allow for efficient storage of node position metadata,
    /// which is updated frequently when user is moving it. Since we're storing full copy
    /// of each metadata version during changes, this can lead to a lot of data being stored
    /// in the history stack. By merging the changes, we can reduce the amount of data stored
    /// and thus improve performance. This also means that when user will be undoing
    /// the move, the node will be moved back to the original position, instead of moving
    /// it back by each step. UI can decide to make some intermediate steps when
    /// there is a lot of moves to have snapshots, but not each small change has to be
    /// stored in the history stack, and this boolean argument allows to control this.
    pub fn alter_node_meta(
        &mut self,
        node_id: canvas::Id,
        key: JsString,
        new: JsValue,
        flatten: bool,
    ) -> Result<(), NodeNotFoundError> {
        trace!("alter node {node_id} metadata for key `{key}`");

        if let Some(last_op) = self.current_change_mut() {
            if flatten {
                if let ChangeOp::AlterNodeMetadata {
                    node_id: last_node_id,
                    key: ref last_key,
                    new: _,
                    backup: ref last_backup,
                } = last_op.op
                {
                    if last_node_id == node_id && *last_key == key {
                        trace!("flatten metadata change for node {node_id} key `{key}`");
                        let new_op = ChangeOp::AlterNodeMetadata {
                            node_id,
                            key,
                            new,
                            backup: last_backup.clone(),
                        };
                        last_op.replace_change(new_op);
                        return Ok(());
                    }
                }
            }
        }

        let op = ChangeOp::alter_node_metadata(self, node_id, key, new)?;
        self.changes.push_op(op);
        Ok(())
    }

    /// Get the known data type of the pin. [None] if the pin type is not known.
    pub fn pin_data_type(&self, pin: canvas::Pin) -> Option<canvas::PrimitiveType> {
        trace!("get data type of pin {pin}");
        self.project.canvas().pin_type(pin).ok().flatten()
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
        trace!("switch current project to {}", project.uuid);
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
            warn!("fake server mode always returns false on `isServerSaved`");
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
            warn!("fake server mode always throws an error on `saveToServer`");
            return Err(JsError::new(
                "Fake server mode throws an error on save attempt",
            ));
        }

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
    #[wasm_bindgen(js_name = changeAt)]
    pub fn change_at(pos: usize) -> Option<JsChangeItem> {
        let ws = work_session().read().expect(WORK_SESSION_POISONED);
        let project = ws.current_project()?;
        project.change_at(pos).map(|change| JsChangeItem {
            timestamp: change.micros_since_unix() as u64,
            position: pos,
            project_uuid: project.uuid(),
        })
    }

    /// Length of the history stack of changes in the current project, disregarding
    /// the current position.
    #[wasm_bindgen(js_name = length, getter)]
    pub fn length() -> usize {
        let ws = work_session().read().expect(WORK_SESSION_POISONED);
        let project = ws.current_project();
        project.map_or(0, |p| p.changes.stack.len())
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
                debug!("js change item - change not found in the project");
                false
            }
        } else {
            debug!("js change item - project not found in the work session");
            false
        }
    }

    /// Get the operation that was performed.
    /// This information is necessary to reflect the change in the UI.
    /// If the handle is invalid, this will return an error.
    pub fn operation(&self) -> Result<JsChangeOp, InvalidHandleError> {
        if !self.is_valid() {
            error!("js change item - use of invalid handle");
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

#[wasm_bindgen(js_class = ChangeOp)]
impl JsChangeOp {
    // TODO

    /// Set metadata for this history item. This allows UI to associate some
    /// additional information with the change, and removes the responsibility of
    /// managing the lifecycle of the metadata from the caller. E.g. if this history
    /// item gets dropped, associated metadata will be dropped automatically as well, without
    /// any JS intervention.
    #[wasm_bindgen(js_name = meta, getter)]
    pub fn set_meta(&mut self, js: JsValue) {
        // TODO
    }

    /// Get metadata associated with this history item.
    /// See [JsChangeOp::set_meta] for more information.
    #[wasm_bindgen(js_name = meta, getter)]
    pub fn get_meta(&self) -> JsValue {
        // TODO
        JsValue::NULL
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
    /// Stack of changes made to the project.
    stack: Vec<ChangeItem>,

    /// Current position in the history stack, that is reflected on the canvas
    /// of the project and, consequently, on the UI.
    pos: usize,
}

impl ChangeStack {
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            pos: 0,
        }
    }

    pub fn push_op(&mut self, op: impl Into<ChangeOp>) {
        // Discard all changes after the current position.
        // There can be some if we had "undone".
        self.stack.truncate(self.pos);

        self.stack.push(ChangeItem::new(op.into()));
        self.pos += 1;
    }
}

/// Checkout changes in the stack of changes. This is used to detect if the project
/// was changed since some state.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CheckoutChangedStack {
    /// Position in the history stack.
    pos: usize,

    /// Timestamp of the last change.
    time: std::time::SystemTime,
}

/// Change item in the history stack of changes. This has one operation
/// associated with it, that can be applied to or reverted in the project.
/// It also has metadata associated with the change from JS side.
/// This is a smallest unit of action recordable in the history stack.
#[derive(Debug)]
pub struct ChangeItem {
    /// Timestamp of the change.
    timestamp: std::time::SystemTime,

    /// Operation that describes this change.
    op: ChangeOp,

    /// Metadata associated with the change from UI. This allows to
    /// manage the lifecycle of the associated value with the history item
    /// itself, so that JS side is not responsible for resource management in case
    /// this history item gets dropped.
    meta: JsValue,
}

impl ChangeItem {
    pub fn new(op: ChangeOp) -> Self {
        Self {
            timestamp: SystemTime::now(),
            op,
            meta: JsValue::UNDEFINED,
        }
    }

    /// Get the timestamp of the change in microseconds since UNIX epoch.
    pub fn micros_since_unix(&self) -> u128 {
        self.timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros()
    }

    /// Replace the change with the given one, and update the timestamp.
    ///
    /// This effectively changes all but the [meta](ChangeItem::meta) of the change item.
    fn replace_change(&mut self, op: ChangeOp) {
        self.op = op;
        self.timestamp = SystemTime::now();
    }
}
