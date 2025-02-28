use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
};

use base::{canvas, table_data};
use chrono::Datelike;
use js_sys::{JsString, RegExp, Uint8Array};
use log::error;
use smallvec::SmallVec;
use thiserror::Error;
use uuid::Uuid;

use wasm_bindgen::prelude::*;

use crate::{
    InvalidHandleError, MyUuid, PermissionError,
    work_session::{self, CheckoutChangedStack, WorkSessionProject, wsr, wsw},
};

/// JavaScript node structures.
mod js_nodes;
pub use js_nodes::*;

pub struct Project {
    name: JsString,
    canvas: canvas::Canvas<Metadata>,
    data: SmallVec<[table_data::Table; 1]>,
    files: SmallVec<[File; 1]>,
    uuid: Uuid,
    pub(crate) meta: Metadata,
}

impl Project {
    pub fn zero() -> Self {
        Self {
            name: JsString::from("Unnamed"),
            canvas: canvas::Canvas::new(),
            data: SmallVec::new(),
            files: SmallVec::new(),
            uuid: Uuid::nil(),
            meta: Metadata::default(),
        }
    }

    /// See [JsProject::get_name].
    pub fn name(&self) -> &JsString {
        &self.name
    }

    pub fn data(&self) -> &[table_data::Table] {
        &self.data
    }

    /// See [JsProject::get_files].
    pub fn files(&self) -> &[File] {
        &self.files
    }

    pub fn file_by_id(&self, uuid: Uuid) -> Option<&File> {
        self.files.iter().find(|file| file.uuid() == uuid)
    }

    /// Add a file to the project. Returns the UUID of the file in this project.
    pub fn add_file(&mut self, mut file: File) -> Uuid {
        // We retry 10 times to generate a unique UUID for the file.
        // If we still fail, we panic.
        for _ in 0..10 {
            let uuid = Uuid::new_v4();
            if self.files.iter().all(|f| f.uuid() != uuid) {
                file.uuid = uuid;
                self.files.push(file);
                return uuid;
            }
        }
        panic!("Failed to generate a unique UUID for the file");
    }

    /// Project UUID.
    pub fn uuid(&self) -> Uuid {
        self.uuid
    }

    pub(crate) fn canvas(&self) -> &canvas::Canvas<Metadata> {
        &self.canvas
    }

    pub(crate) fn canvas_mut(&mut self) -> &mut canvas::Canvas<Metadata> {
        &mut self.canvas
    }
}

/// A project handle that can be used to access project data from the current work session.
#[wasm_bindgen(js_name = Project)]
pub struct JsProject {
    pub(crate) uuid: Uuid,
}

#[wasm_bindgen(js_class = Project)]
impl JsProject {
    /// Get the files in the project. Note that some of the files can be declared
    /// but contents not actually loaded into the memory. This may be the case if the file is
    /// still being loaded or if the user has no permission to load the file content.
    /// However, this list will contain all files that the project has with associated metadata
    /// as long as the user has permission to view the project at least read-only.
    #[wasm_bindgen(getter, js_name = files)]
    pub fn get_files(&self) -> Vec<JsProjectFile> {
        let ws = wsr!();
        let project = if let Some(project) = ws.project_by_id(self.uuid) {
            project
        } else {
            error!("Project not found. Was removed from work session. Project handle is invalid.");
            return Vec::new();
        };

        project
            .files()
            .iter()
            .map(|file| JsProjectFile {
                name: file.name().clone(),
                uuid: file.uuid().into_js_array(),
                project_uuid: Some(project.uuid()),
                is_loaded: file.bytes().is_some(),
                can_load: file.can_load(),
                protect: JsProtect::from(file.protect()),
            })
            .collect()
    }

    /// Get the name of the project.
    #[wasm_bindgen(getter, js_name = name)]
    pub fn get_name(&self) -> JsString {
        let ws = wsr!();
        let project = ws.project_by_id(self.uuid);
        if let Some(project) = project {
            project.name().clone()
        } else {
            error!("{ProjectNotFoundError}");
            JsString::from("")
        }
    }

    /// Set the new name for the project.
    #[wasm_bindgen(setter, js_name = name)]
    pub fn set_name(&mut self, name: JsString) {
        let mut ws = wsw!();
        let project = ws.project_by_id_mut(self.uuid);
        if let Some(project) = project {
            project.name = name;
        } else {
            error!("{ProjectNotFoundError}");
        }
    }

    /// Load the project with the given identifier, as returned by listing request.
    ///
    /// # Errors
    /// This will throw ProjectLoadError if the project cannot be loaded, or already exists
    /// in the current work session.
    ///
    /// # Fake Server
    /// UUID 0 allows to load empty project as if it was loaded from the server.
    pub async fn load(identifier: JsString) -> Result<JsProject, ProjectLoadError> {
        let identifier = String::from(identifier);
        let uuid = Uuid::parse_str(&identifier).map_err(|_| ProjectLoadError::InvalidUuid)?;

        if cfg!(feature = "fake_server") {
            if uuid == Uuid::nil() {
                let mut ws = wsw!();
                ws.add_project(Project::zero())?;
                Ok(JsProject { uuid })
            } else {
                error!("Failed to load fake server project by identifier `{identifier}`");
                Err(ProjectLoadError::NotFound)
            }
        } else {
            todo!()
        }
    }

    /// Canvas of the project. This is where the nodes are placed.
    #[wasm_bindgen(getter)]
    pub fn canvas(&mut self) -> JsCanvas {
        JsCanvas {
            project_uuid: self.uuid,
        }
    }

    /// Get the metadata associated with the project by the client.
    #[wasm_bindgen(getter)]
    pub fn meta(&self) -> JsProjectMeta {
        JsProjectMeta {
            project_uuid: self.uuid,
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = ProjectLoadError)]
pub enum ProjectLoadError {
    /// Given UUID cannot be parsed or is invalid.
    InvalidUuid,
    AlreadyLoaded,
    NotFound,
    PermissionError,
    RequestError,
    ServerError,
}

impl From<work_session::ProjectExistsError> for ProjectLoadError {
    fn from(_: work_session::ProjectExistsError) -> Self {
        ProjectLoadError::AlreadyLoaded
    }
}

/// Canvas of the project. This is where the nodes are placed.
#[derive(Debug, Copy, Clone)]
#[wasm_bindgen(js_name = Canvas)]
pub struct JsCanvas {
    project_uuid: Uuid,
}

/// Error when the project is not found in the work session by the given identifier.
/// This can happen if the handle to the project is invalid because the project was
/// removed from the work session.
#[derive(Debug, Error)]
#[wasm_bindgen]
#[error("Project not found. Was removed from work session. Project handle is invalid.")]
pub struct ProjectNotFoundError;

impl From<ProjectNotFoundError> for InvalidHandleError {
    fn from(_: ProjectNotFoundError) -> Self {
        InvalidHandleError
    }
}

/// A trait for handles that are associated with a project.
pub(crate) trait ProjectHandle {
    fn project_uuid(&self) -> Uuid;

    /// Validate the handle by some inner custom mechanism.
    /// This is used to ensure that the handle is still valid even when Project UUID might
    /// be valid by itself, unlike some other inner resource the handle points to.
    fn inner_validation(&self) -> bool {
        true
    }

    /// Check the validity of the handle and then proceed with the given function.
    /// If the project is not found, this will return an [ProjectNotFoundError].
    #[track_caller]
    fn checked_write<T>(
        &self,
        f: impl FnOnce(&mut WorkSessionProject) -> Result<T, JsError>,
    ) -> Result<T, JsError> {
        if !self.inner_validation() {
            return Err(InvalidHandleError.into());
        }

        let mut ws = wsw!();
        let project = ws.project_by_id_mut(self.project_uuid());
        if let Some(project) = project {
            f(project)
        } else {
            let err = ProjectNotFoundError;
            error!("{err}");
            Err(err.into())
        }
    }

    /// Check the validity of the handle and then proceed with the given function.
    /// If the project is not found, this will return an [ProjectNotFoundError].
    #[track_caller]
    fn checked_read<T>(
        &self,
        f: impl FnOnce(&WorkSessionProject) -> Result<T, JsError>,
    ) -> Result<T, JsError> {
        if !self.inner_validation() {
            return Err(InvalidHandleError.into());
        }

        let ws = wsr!();
        let project = ws.project_by_id(self.project_uuid());
        if let Some(project) = project {
            f(project)
        } else {
            let err = ProjectNotFoundError;
            error!("{err}");
            Err(err.into())
        }
    }
}

impl ProjectHandle for JsCanvas {
    fn project_uuid(&self) -> Uuid {
        self.project_uuid
    }
}

#[wasm_bindgen(js_class = Canvas)]
impl JsCanvas {
    /// Add a new node to the canvas.
    #[wasm_bindgen(js_name = addNode)]
    pub fn add_node(&mut self, stub: JsNodeStub) -> Result<JsNode, JsError> {
        self.checked_write(|project| {
            let node_id = project.add_node(stub.stub);
            Ok(JsNode {
                project_uuid: self.project_uuid,
                node_id,
            })
        })
    }

    /// Add a new edge between two pins.
    ///
    /// # Errors
    /// If the edge cannot be added, this will return an error.
    #[wasm_bindgen(js_name = addEdge)]
    pub fn add_edge(&mut self, from: JsNodePin, to: JsNodePin) -> Result<(), JsError> {
        use canvas::{Edge, InputPin, OutputPin, Pin};

        self.checked_write(|project| {
            let from = OutputPin(Pin {
                node_id: from.node_id,
                order: from.ordinal,
            });
            let to = InputPin(Pin {
                node_id: to.node_id,
                order: to.ordinal,
            });
            project.add_edge(Edge { from, to })?;
            Ok(())
        })
    }

    /// Remove an edge between two pins.
    ///
    /// # Errors
    /// If the edge does not exist, this will return an error.
    #[wasm_bindgen(js_name = removeEdge)]
    pub fn remove_edge(&mut self, from: JsNodePin, to: JsNodePin) -> Result<(), JsError> {
        use canvas::{Edge, InputPin, OutputPin, Pin};

        self.checked_write(|project| {
            let from = OutputPin(Pin {
                node_id: from.node_id,
                order: from.ordinal,
            });
            let to = InputPin(Pin {
                node_id: to.node_id,
                order: to.ordinal,
            });
            project.remove_edge(Edge { from, to })?;
            Ok(())
        })
    }

    /// Get iterator over nodes in this canvas.
    ///
    /// This is useful when you want to iterate over all nodes in the canvas, e.g. to
    /// initialize UI view with all nodes.
    ///
    /// This iterator is broken when the canvas is modified, and
    /// may not return all nodes or have some nodes repeated in such case. You
    /// should not have this iterator around while modifying the canvas.
    #[wasm_bindgen(js_name = nodeIterator)]
    pub fn node_iter(&self) -> Result<JsNodeIter, JsError> {
        self.checked_read(|project| {
            Ok(JsNodeIter {
                stack_checkout: project.stack_checkout(),
                project_uuid: self.project_uuid,
                pos: 0,
            })
        })
    }
}

/// Node in the canvas. This is actual instance of the placed node in a project.
#[derive(Debug)]
#[wasm_bindgen(js_name = Node)]
pub struct JsNode {
    project_uuid: Uuid,
    node_id: canvas::Id,
}

impl ProjectHandle for JsNode {
    fn project_uuid(&self) -> Uuid {
        self.project_uuid
    }
}

/// A trait for handles that are associated with a node.
pub(crate) trait NodeHandle: ProjectHandle {
    fn node_id(&self) -> canvas::Id;

    /// Check the validity of the node handle and then proceed with the given function.
    /// If the project is not found, this will return an [ProjectNotFoundError].
    /// If the node is not found, this will return an [InvalidHandleError].
    fn checked_node_read<T>(
        &self,
        f: impl FnOnce(&canvas::Node<Metadata>) -> Result<T, JsError>,
    ) -> Result<T, JsError> {
        let node_id = self.node_id();
        self.checked_read(|project| {
            let node = project.canvas().node(node_id).ok_or(InvalidHandleError)?;
            f(node)
        })
    }

    /// Check the validity of the node handle, ensuring node and project exist,
    /// and then proceed with the given function.
    /// If the project is not found, this will return an [ProjectNotFoundError].
    /// If the node is not found, this will return an [InvalidHandleError].
    fn ensure_node_write<T>(
        &self,
        f: impl FnOnce(&mut WorkSessionProject) -> Result<T, JsError>,
    ) -> Result<T, JsError> {
        let node_id = self.node_id();
        self.checked_write(|project| {
            project
                .canvas_mut()
                .node(node_id)
                .ok_or(InvalidHandleError)?;
            f(project)
        })
    }
}

impl NodeHandle for JsNode {
    fn node_id(&self) -> canvas::Id {
        self.node_id
    }
}

#[wasm_bindgen(js_class = Node)]
impl JsNode {
    /// Get the stub that will allow to create a new node with the same configuration.
    ///
    /// # Errors
    /// If the project or the node is no longer valid, this will return an error.
    #[wasm_bindgen(getter, js_name = stub)]
    pub fn stub(&self) -> Result<JsNodeStub, JsError> {
        self.checked_node_read(|node| Ok(node.stub.clone().into()))
    }

    /// Get the output pin at the given position.
    #[wasm_bindgen(js_name = outAt)]
    pub fn out_at(&self, position: u32) -> Result<JsNodePin, JsError> {
        self.checked_node_read(|node| {
            if let Some(ordinal) = node.stub.real_output_pin_idx(position as _) {
                Ok(JsNodePin {
                    project_uuid: self.project_uuid,
                    node_id: self.node_id,
                    ordinal: ordinal as _,
                })
            } else {
                error!("Output pin not found at position {position}");
                Err(PinNotFoundError.into())
            }
        })
    }

    /// Get the input pin at the given position.
    #[wasm_bindgen(js_name = inAt)]
    pub fn in_at(&self, position: u32) -> Result<JsNodePin, JsError> {
        self.checked_node_read(|node| {
            let ordinal = node
                .stub
                .real_input_pin_idx(position as _)
                .ok_or(PinNotFoundError)?;
            Ok(JsNodePin {
                project_uuid: self.project_uuid,
                node_id: self.node_id,
                ordinal,
            })
        })
    }

    /// Get the node identifier.
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> u32 {
        self.node_id.get()
    }

    /// Drop the node from the canvas. You should not use the node handle after this.
    /// All edges connected to this node will be removed, and stored metadata
    /// removed.
    ///
    /// This also returns the stub of the node, so you can recreate it later.
    pub fn drop(self) -> Result<JsNodeStub, JsError> {
        let stub = self.stub()?;
        self.checked_write(|project| {
            project.remove_node(self.node_id)?;
            Ok(stub)
        })
    }

    /// Get the metadata associated with the node by the client.
    #[wasm_bindgen(getter)]
    pub fn meta(&self) -> JsNodeMeta {
        JsNodeMeta {
            project_uuid: self.project_uuid,
            node_id: self.node_id,
        }
    }
}

/// Node iterator to fetch nodes from the canvas. This exists so to avoid
/// duplicating big arrays of nodes in memory.
#[derive(Debug)]
#[wasm_bindgen(js_name = NodeIter)]
pub struct JsNodeIter {
    stack_checkout: CheckoutChangedStack,

    project_uuid: Uuid,
    pos: usize,
}

impl ProjectHandle for JsNodeIter {
    fn project_uuid(&self) -> Uuid {
        self.project_uuid
    }
}

impl NodeHandle for JsNodeIter {
    fn node_id(&self) -> canvas::Id {
        self.checked_read(|p| {
            p.canvas()
                .node_by_idx(self.pos as _)
                .ok_or(JsError::new("Node not found"))
                .map(|v| v.id)
        })
        .unwrap_or(canvas::Id::MAX)
    }
}

#[wasm_bindgen(js_class = NodeIter)]
impl JsNodeIter {
    /// Get the next node in the canvas.
    /// Returns `undefined` if there are no more nodes.
    pub fn next(&mut self) -> Option<JsNode> {
        if !self.is_valid() {
            error!("Node iterator is invalid. Stopping iteration.");
            return None;
        }

        let node_id = self.node_id();
        if !node_id.is_valid() {
            return None;
        }

        self.pos += 1;
        Some(JsNode {
            project_uuid: self.project_uuid,
            node_id,
        })
    }

    #[wasm_bindgen(getter, js_name = isValid)]
    pub fn is_valid(&self) -> bool {
        self.checked_read(|p| Ok(p.stack_checkout() == self.stack_checkout))
            .unwrap_or(false)
    }
}

/// Node pin. This is a connection point on a node.
#[derive(Debug)]
#[wasm_bindgen(js_name = NodePin)]
pub struct JsNodePin {
    project_uuid: Uuid,
    node_id: canvas::Id,
    ordinal: canvas::PinOrder,
}

impl ProjectHandle for JsNodePin {
    fn project_uuid(&self) -> Uuid {
        self.project_uuid
    }
}

impl NodeHandle for JsNodePin {
    fn node_id(&self) -> canvas::Id {
        self.node_id
    }
}

#[wasm_bindgen(js_class = NodePin)]
impl JsNodePin {
    /// Get the data type of the pin.
    /// If the pin has unknown data type, this will return `undefined`.
    #[wasm_bindgen(getter, js_name = dataType)]
    pub fn data_type(&self) -> Result<Option<JsDataType>, JsError> {
        self.checked_read(|project| {
            let ty = project
                .pin_data_type(canvas::Pin {
                    node_id: self.node_id,
                    order: self.ordinal,
                })
                .map(|v| v.to_owned().into());
            Ok(ty)
        })
    }

    /// Whether the pin is an output pin (`true`). If it is an input pin, this returns `false`.
    #[wasm_bindgen(getter, js_name = isOutput)]
    pub fn is_output(&self) -> Result<bool, JsError> {
        self.checked_node_read(|node| {
            if node.stub.is_valid_input_ordinal(self.ordinal) {
                Ok(false)
            } else if node.stub.is_valid_output_ordinal(self.ordinal) {
                Ok(true)
            } else {
                error!("Pin ordinal is invalid. Cannot determine if it is input or output.");
                Err(PinNotFoundError.into())
            }
        })
    }

    /// Get the position of the pin on the node.
    ///
    /// This is a converted ordinal, not absolute position in the node as per internal logic.
    /// This uniquely identifies input and output pins separately.
    #[wasm_bindgen(getter)]
    pub fn ordinal(&self) -> Result<canvas::PinOrder, JsError> {
        let is_output = self.is_output()?;
        self.checked_node_read(|node| {
            if is_output {
                node.stub.real_output_pin_idx(self.ordinal)
            } else {
                node.stub.real_input_pin_idx(self.ordinal)
            }
            .ok_or(PinNotFoundError.into())
        })
    }

    /// Peek into the data that this pin receiver or transmits.
    /// This may be useful for UI to show the data for user debugging or analysis.
    #[wasm_bindgen(js_name = peekInto)]
    pub fn peek_into(&self) -> JsPeekFlow {
        todo!()
    }

    /// Get the errors that this pin is associated with.
    #[wasm_bindgen(getter)]
    pub fn errors(&self) -> Vec<JsNodePinError> {
        todo!()
    }
}

#[derive(Debug, Error)]
#[wasm_bindgen]
#[error("described pin cannot be found in the node")]
pub struct PinNotFoundError;

/// Internal struct representing metadata. This metadata can be assigned to nodes, projects,
/// history items etc...
///
/// # Implementation Note
/// We split this into key-value structure as to allow updating little pieces
/// of metadata. Otherwise, we would had to update the whole metadata object,
/// which can be big and hence operation would be inefficient.
#[derive(Debug, Default, Clone)]
pub struct Metadata {
    map: HashMap<HashJsString, JsValue>,
}

/// Hashable version of the JavaScript string for Rust.
/// This is used as a key in the metadata map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashJsString(JsString);

impl std::hash::Hash for HashJsString {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for code in self.0.iter() {
            code.hash(state);
        }
    }
}

impl From<JsString> for HashJsString {
    fn from(s: JsString) -> Self {
        Self(s)
    }
}

impl From<HashJsString> for JsString {
    fn from(s: HashJsString) -> Self {
        s.0
    }
}

impl Deref for HashJsString {
    type Target = JsString;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for Metadata {
    type Target = HashMap<HashJsString, JsValue>;

    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

impl DerefMut for Metadata {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.map
    }
}

/// Node metadata. This is a key-value map of metadata associated with the node.
/// This can be used to store additional information about the node for UI purposes.
/// This is not used by the canvas itself.
#[derive(Debug, Clone)]
#[wasm_bindgen(js_name = NodeMeta)]
pub struct JsNodeMeta {
    project_uuid: Uuid,
    node_id: canvas::Id,
}

impl ProjectHandle for JsNodeMeta {
    fn project_uuid(&self) -> Uuid {
        self.project_uuid
    }
}

impl NodeHandle for JsNodeMeta {
    fn node_id(&self) -> canvas::Id {
        self.node_id
    }
}

/// Key-value map of metadata associated with the project.
/// Similar to [JsNodeMeta] for nodes.
#[derive(Debug, Clone)]
#[wasm_bindgen(js_name = ProjectMeta)]
pub struct JsProjectMeta {
    project_uuid: Uuid,
}

impl ProjectHandle for JsProjectMeta {
    fn project_uuid(&self) -> Uuid {
        self.project_uuid
    }
}

/// JavaScript metadata trait. So that to conform to a single interface to reuse
/// in similar structures.
pub trait JsMeta {
    /// Execute given closure over the reference to the metadata.
    fn meta<T>(&self, f: impl FnOnce(&Metadata) -> Result<T, JsError>) -> Result<T, JsError>;

    /// Mutable reference to the metadata associated with the object.
    fn meta_mut<T>(
        &mut self,
        f: impl FnOnce(&mut Metadata) -> Result<T, JsError>,
    ) -> Result<T, JsError>;

    /// Get the value associated with the given key.
    fn get(&self, key: JsString) -> Result<JsValue, JsError> {
        self.meta(|meta| Ok(meta.get(&key.into()).cloned().unwrap_or(JsValue::UNDEFINED)))
    }

    /// Set the value associated with the given key, removing the old value if it exists.
    /// Returns the removed value that was assigned beforehand (if any).
    fn set(&mut self, key: JsString, value: JsValue) -> Result<JsValue, JsError> {
        self.meta_mut(|meta| {
            Ok(meta
                .insert(key.into(), value.into())
                .unwrap_or(JsValue::UNDEFINED)
                .into())
        })
    }

    /// Remove the value associated with the given key. Returns the value that was removed,
    /// if it was assigned beforehand.
    fn remove(&mut self, key: JsString) -> Result<JsValue, JsError> {
        self.meta_mut(|meta| {
            Ok(meta
                .remove(&key.into())
                .unwrap_or(JsValue::UNDEFINED)
                .into())
        })
    }

    /// Get the keys of the metadata.
    fn keys(&self) -> Result<Vec<JsString>, JsError> {
        self.meta(|meta| Ok(meta.keys().cloned().map(Into::into).collect()))
    }
}

impl JsMeta for JsNodeMeta {
    fn meta<T>(&self, f: impl FnOnce(&Metadata) -> Result<T, JsError>) -> Result<T, JsError> {
        self.checked_node_read(|node| f(&node.meta))
    }

    fn meta_mut<T>(
        &mut self,
        f: impl FnOnce(&mut Metadata) -> Result<T, JsError>,
    ) -> Result<T, JsError> {
        self.ensure_node_write(|node| f(&mut node.meta))
    }
}

impl JsMeta for JsProjectMeta {
    fn meta<T>(&self, f: impl FnOnce(&Metadata) -> Result<T, JsError>) -> Result<T, JsError> {
        self.checked_write(|project| f(&project.meta))
    }

    fn meta_mut<T>(
        &mut self,
        f: impl FnOnce(&mut Metadata) -> Result<T, JsError>,
    ) -> Result<T, JsError> {
        self.checked_write(|project| f(&mut project.meta))
    }
}

/// Kind of the data type.
#[derive(Debug, Clone, Copy)]
#[wasm_bindgen(js_name = DataTypeKind)]
pub enum JsDataTypeKind {
    Int,
    Uint,
    Unit,
    Moneraty,
    Date,
    DateTime,
    Time,
    Bool,
    Str,
    Ordering,
    File,
    Record,
    Array,
    Predicate,
    Result,
    Option,
}

#[derive(Debug, Clone)]
#[wasm_bindgen(js_name = DataType)]
pub struct JsDataType {
    repr: canvas::PrimitiveType,
}

#[wasm_bindgen(js_class = DataType)]
impl JsDataType {
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> JsDataTypeKind {
        use JsDataTypeKind as J;
        use canvas::PrimitiveType::*;
        match &self.repr {
            Int => J::Int,
            Uint => J::Uint,
            Unit => J::Unit,
            Moneraty => J::Moneraty,
            Date => J::Date,
            DateTime => J::DateTime,
            Time => J::Time,
            Bool => J::Bool,
            Str => J::Str,
            Ordering => J::Ordering,
            File => J::File,
            Record => J::Record,
            Array(_) => J::Array,
            Result(_) => J::Result,
            Option(_) => J::Option,
        }
    }
}

impl From<canvas::PrimitiveType> for JsDataType {
    fn from(repr: canvas::PrimitiveType) -> Self {
        Self { repr }
    }
}

impl From<JsDataType> for canvas::PrimitiveType {
    fn from(js: JsDataType) -> Self {
        js.repr
    }
}

/// A value as an instance of a data type supported by the canvas.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
#[wasm_bindgen(js_name = Value)]
pub struct JsDataInstance {
    value: canvas::Value,
}

#[wasm_bindgen(js_class = Value)]
impl JsDataInstance {
    /// Get the data type of the value.
    #[wasm_bindgen(getter, js_name = dataType)]
    pub fn data_type(&self) -> JsDataType {
        self.value.type_of().into()
    }

    /// Get the value converted as a JS value.
    /// Note this conversion is lossy, e.g. monetary values are converted
    /// into floating point number in JavaScript, but in fact it is stored
    /// as a decimal in the canvas (hence is lossless).
    #[wasm_bindgen(getter)]
    pub fn value(&self) -> JsValue {
        use bigdecimal::ToPrimitive;
        use canvas::Value::*;
        use chrono::Timelike;
        use std::cmp::Ordering;
        use std::ops::Deref;
        match &self.value {
            Int(i) => JsValue::from(*i),
            Uint(i) => JsValue::from(*i),
            Unit => JsValue::null(),
            Moneraty(monetary) => JsMonetary {
                amount: monetary.to_f64().unwrap(),
            }
            .into(),
            Date(naive_date) => js_sys::Date::new_with_year_month_day(
                naive_date.year() as _,
                naive_date.month() as _,
                naive_date.day() as _,
            )
            .into(),
            DateTime(naive_date_time) => js_sys::Date::new_with_year_month_day_hr_min_sec(
                naive_date_time.year() as _,
                naive_date_time.month() as _,
                naive_date_time.day() as _,
                naive_date_time.hour() as _,
                naive_date_time.minute() as _,
                naive_date_time.second() as _,
            )
            .into(),
            Time(time) => JsTime {
                hour: time.hour() as _,
                minute: time.minute() as _,
                second: time.second() as _,
            }
            .into(),
            Bool(v) => JsValue::from(*v),
            Str(compact_string) => JsValue::from(compact_string.as_str()),
            Ordering(ordering) => match ordering {
                Ordering::Less => JsOrdering::Less,
                Ordering::Equal => JsOrdering::Equal,
                Ordering::Greater => JsOrdering::Greater,
            }
            .into(),
            Array(values) => {
                let arr = js_sys::Array::new_with_length(values.len() as u32);
                for (i, value) in values.iter().enumerate() {
                    arr.set(
                        i as u32,
                        JsDataInstance {
                            value: value.clone(),
                        }
                        .into(),
                    );
                }
                arr.into()
            }
            Result { value, .. } => match value {
                Ok(value) => JsResult {
                    value: JsDataInstance {
                        value: value.deref().clone(),
                    }
                    .into(),
                    is_ok: true,
                },
                Err(value) => JsResult {
                    value: JsDataInstance {
                        value: value.deref().clone(),
                    }
                    .into(),
                    is_ok: false,
                },
            }
            .into(),
            Option { value, .. } => JsOption {
                value: value.as_deref().map(|value| JsDataInstance {
                    value: value.to_owned(),
                }),
            }
            .into(),
        }
    }
}

/// Time representation for JS.
#[derive(Debug)]
#[wasm_bindgen(js_name = Time)]
pub struct JsTime {
    pub hour: i8,
    pub minute: i8,
    pub second: i8,
}

/// Monetary approximation for JavaScript.
#[derive(Debug)]
#[wasm_bindgen(js_name = Monetary)]
pub struct JsMonetary {
    pub amount: f64,
}

/// Ordering representation for JS.
#[derive(Debug)]
#[wasm_bindgen(js_name = Ordering)]
pub enum JsOrdering {
    Less,
    Equal,
    Greater,
}

/// Result representation for JS.
#[derive(Debug)]
#[wasm_bindgen(js_name = Result)]
pub struct JsResult {
    /// The value of the result (either Ok or Err).
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub value: JsDataInstance,

    /// Whether the result is Ok.
    #[wasm_bindgen(readonly, js_name = isOk)]
    pub is_ok: bool,
}

/// Option representation for JS.
#[derive(Debug)]
#[wasm_bindgen(js_name = Option)]
pub struct JsOption {
    /// The value of the option, if it is Some.
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub value: Option<JsDataInstance>,
}

/// Predicate representation for JS.
#[derive(Debug)]
#[wasm_bindgen(js_name = Predicate)]
pub struct JsPredicate {
    /// Listed input pin types for the predicate.
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub inputs: Vec<JsDataType>,

    /// Listed output pin types for the predicate.
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub outputs: Vec<JsDataType>,
}

/// Peek into the flow of values. This is useful for debugging and analysis.
#[derive(Debug)]
#[wasm_bindgen(js_name = PeekFlow)]
pub struct JsPeekFlow {
    // TODO
}

#[wasm_bindgen(js_class = PeekFlow)]
impl JsPeekFlow {
    /// Get the data type of the values in this flow.
    #[wasm_bindgen(js_name = dataType)]
    pub fn data_type(&self) -> JsDataTypePeek {
        todo!()
    }

    /// Shuffle and get the array of unique values, up to the given limit.
    /// Array will have less elements if the flow does not have enough unique values.
    ///
    /// This is implemented in such way so that the example validations are quick
    /// to produce several unique values, omitting full calculation of all test-file provided data.
    /// The process of fetching unique values is stopped as soon as the limit is reached.
    /// It is obviously much faster than to validate full
    /// file with possibly tens of thousands of rows just to show 10 or 20 unique values.
    /// Caller thus can expect almost instant result for small limits even in complex flows.
    #[wasm_bindgen(js_name = shuffled)]
    pub fn shuffled(&self, limit: usize) -> Vec<JsFlowValue> {
        todo!()
    }
}

/// Peek into the data type. This allows to see the nested data types.
#[derive(Debug)]
#[wasm_bindgen(js_name = DataTypePeek)]
pub struct JsDataTypePeek {
    inner: JsDataType,
}

#[wasm_bindgen(js_class = DataTypePeek)]
impl JsDataTypePeek {
    /// Get the kind of the data type.
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> JsDataTypeKind {
        self.inner.kind()
    }

    /// Get the inner data type if this is a nested data type.
    /// Positional argument is the position in the nested data type.
    ///
    /// # Examples
    /// `Result<Result<T1, E1>, Result<T2, E2>>`:
    /// - `inner(0)` returns peek for `Result`.
    /// - `inner(1)` returns `undefined`.
    /// - `inner(0).inner(0)` returns peek for `Result`.
    /// - `inner(0).inner(1)` returns peek for `Result`.
    /// - `inner(0).inner(2)` returns `undefined`.
    /// - `inner(0).inner(0).inner(0)` returns peek for `T1`.
    /// - `inner(0).inner(0).inner(1)` returns peek for `E1`.
    /// - `inner(0).inner(1).inner(0)` returns peek for `T2`.
    /// - `inner(0).inner(1).inner(1)` returns peek for `E2`.
    /// - `inner(0).inner(1).inner(2)` returns `undefined`.
    #[wasm_bindgen()]
    pub fn inner(&self, pos: usize) -> Option<JsDataTypePeek> {
        use canvas::PrimitiveType as PT;
        let repr = match &self.inner.repr {
            PT::Int => return None,
            PT::Uint => return None,
            PT::Unit => return None,
            PT::Moneraty => return None,
            PT::Date => return None,
            PT::DateTime => return None,
            PT::Time => return None,
            PT::Bool => return None,
            PT::Str => return None,
            PT::Ordering => return None,
            PT::File => return None,
            PT::Record => return None,
            PT::Array(primitive_type) => {
                if pos == 0 {
                    (**primitive_type).to_owned()
                } else {
                    return None;
                }
            }
            PT::Result(boxed) => {
                if pos == 0 {
                    (**boxed).0.to_owned()
                } else if pos == 1 {
                    (**boxed).1.to_owned()
                } else {
                    return None;
                }
            }
            PT::Option(primitive_type) => {
                if pos == 0 {
                    (**primitive_type).to_owned()
                } else {
                    return None;
                }
            }
        };
        Some(JsDataTypePeek {
            inner: JsDataType { repr },
        })
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = NodePinError)]
pub struct JsNodePinError {}

/// Value in the flow. This is a peek of a unique value from some flow.
#[derive(Debug)]
#[wasm_bindgen(js_name = FlowValue)]
pub struct JsFlowValue {}

#[wasm_bindgen(js_class = FlowValue)]
impl JsFlowValue {
    /// How many times this value repeats.
    /// This value may not always be accurate, as it is calculated on the fly and
    /// in complex flows it may be too expensive to calculate the exact count.
    /// Assume this is "at least" count.
    #[wasm_bindgen(getter)]
    pub fn count(&self) -> usize {
        todo!()
    }

    /// The value itself.
    #[wasm_bindgen(getter)]
    pub fn value(&self) -> JsValue {
        todo!()
    }
}

/// See [JsProjectFile] for more information.
#[derive(Debug)]
pub struct File {
    name: JsString,

    /// The file's contents. This is `None` if the file is not loaded into memory.
    bytes: Option<Vec<u8>>,

    /// Whether the file can be loaded into memory.
    can_load: bool,

    protect: Protect,

    uuid: Uuid,
}

impl File {
    pub fn name(&self) -> &JsString {
        &self.name
    }

    pub fn bytes(&self) -> Option<&[u8]> {
        self.bytes.as_deref()
    }

    pub fn can_load(&self) -> bool {
        self.can_load || self.bytes.is_some()
    }

    pub fn protect(&self) -> Protect {
        self.protect
    }

    pub fn uuid(&self) -> Uuid {
        self.uuid
    }
}

/// File in a project. This handle points to a file effective in some commit in
/// the history of a project (or in work session with current uncommited changes).
#[derive(Debug)]
#[wasm_bindgen(js_name = ProjectFile)]
pub struct JsProjectFile {
    /// File name. This is the original name selected by the user.
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub name: JsString,

    /// UUID of the file.
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub uuid: Uint8Array,

    /// UUID of the project that this file belongs to.
    /// If this is `None`, the file is not associated with any project.
    project_uuid: Option<Uuid>,

    /// Whether the file is loaded into the memory.
    is_loaded: bool,

    /// Whether the file can be loaded into the memory.
    can_load: bool,

    /// Protection levels for the file.
    #[wasm_bindgen(readonly)]
    pub protect: JsProtect,
}

#[wasm_bindgen(js_class = ProjectFile)]
impl JsProjectFile {
    /// Whether the file can be loaded into the memory.
    /// This is `true` if the file is already loaded. Even if
    /// the file is not stored on the server, this will be `true` as long
    /// as it remains in the memory.
    #[wasm_bindgen(getter = canLoad)]
    pub fn can_load(&self) -> bool {
        if self.project_uuid.is_none() {
            false
        } else {
            self.can_load
        }
    }

    /// Whether the file is loaded into the memory.
    #[wasm_bindgen(getter = isLoaded)]
    pub fn is_loaded(&self) -> bool {
        if self.project_uuid.is_none() {
            false
        } else {
            self.is_loaded
        }
    }

    /// Load the file into the memory from the server.
    /// This will throw FileLoadError if the file fails to be loaded.
    pub async fn load(&self) -> Result<(), FileLoadError> {
        if cfg!(feature = "fake_server") {
            return Err(FileLoadError::ServerError);
        }

        todo!()
    }

    /// Remove the file from the associated project.
    /// This will throw PermissionError if the user does not have permission to delete the file.
    /// Drop is effective on project commit. Note that this does not remove
    /// the file from previous commits.
    pub fn drop(&self) -> Result<(), PermissionError> {
        todo!()
    }
}

#[derive(Debug, Clone, Copy)]
#[wasm_bindgen(js_name = FileLoadError)]
pub enum FileLoadError {
    /// File was dropped from the project and cannot be loaded.
    Dropped,

    /// File is not stored on the server.
    NotStored,

    /// Insufficient permissions to load the file content.
    InsufficientPermissions,

    /// Server failed to respond.
    ServerError,

    /// Request failed to be sent.
    RequestError,
}

/// See [JsProtect] for more information.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Protect {
    read: ProtectLevel,

    /// Whether the resource can be written to.
    /// Transitively, this implies that the resource can be deleted.
    write: ProtectLevel,
}

impl Protect {
    pub fn new(read: ProtectLevel, write: ProtectLevel) -> Self {
        Self { read, write }
    }

    pub fn read(&self) -> ProtectLevel {
        self.read
    }

    pub fn write(&self) -> ProtectLevel {
        self.write
    }

    pub fn delete(&self) -> ProtectLevel {
        self.write
    }
}

/// Protection configuration for a resource.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[wasm_bindgen(js_name = Protect)]
pub struct JsProtect {
    read: ProtectLevel,
    write: ProtectLevel,
}

#[wasm_bindgen(js_class = Protect)]
impl JsProtect {
    /// Protection level for reading the resource.
    #[wasm_bindgen(getter)]
    pub fn read(&self) -> ProtectLevel {
        self.read
    }

    /// Protection level for writing to the resource.
    #[wasm_bindgen(getter)]
    pub fn write(&self) -> ProtectLevel {
        self.write
    }

    /// Protection level for deleting the resource.
    #[wasm_bindgen(getter)]
    pub fn delete(&self) -> ProtectLevel {
        self.write
    }
}

impl From<Protect> for JsProtect {
    fn from(protect: Protect) -> Self {
        Self {
            read: protect.read,
            write: protect.write,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[wasm_bindgen]
pub enum ProtectLevel {
    /// Owner and higher ranked users are permitted (employers, trusts, admins...).
    HigherRank,

    /// No restrictions.
    ///
    /// More restrictions can still apply transitively e.g.
    /// if project itself is protected
    /// then child resource is protected. Such transitivety does not make sense
    /// for normal front-end usecase (user won't see files in project, as they cannot open it),
    /// but this describes back-end security logic, which should forbid direct requests for files
    /// as well as project.
    Relax,
}

#[derive(Debug)]
pub struct FileBuilder {
    name: JsString,
    protect_read: ProtectLevel,
    protect_delete: ProtectLevel,
    bytes: Uint8Array,
}

impl FileBuilder {
    pub fn protect(&self) -> Protect {
        Protect::new(self.protect_read, self.protect_delete)
    }
}

#[wasm_bindgen(js_name = FileBuilder)]
pub struct JsFileBuilder {
    /// File name. This is the original name selected by the user.
    /// Normally, it is set when the file is being uploaded, but can be changed
    /// before actually saving the file to a project.
    #[wasm_bindgen(getter_with_clone)]
    pub name: Option<JsString>,

    /// Protection level for reading the file.
    #[wasm_bindgen(js_name = protectRead)]
    pub protect_read: Option<ProtectLevel>,

    /// Protection level for deleting the file.
    #[wasm_bindgen(js_name = protectDelete)]
    pub protect_delete: Option<ProtectLevel>,

    /// File contents as a JS byte array.
    bytes: Option<Uint8Array>,
}

#[wasm_bindgen(js_class = FileBuilder)]
impl JsFileBuilder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            name: None,
            protect_read: None,
            protect_delete: None,
            bytes: None,
        }
    }

    /// Set the file contents from a JS file object.
    #[wasm_bindgen(setter, js_name = file)]
    pub fn set_file(&mut self, file: crate::File) {
        self.name = Some(file.name());
        self.bytes = Some(file.bytes());
    }

    /// Build the file with the configured properties into the given project.
    /// FileBuilder handle should not be used after this call (it is consumed).
    ///
    /// Throws FileBuilderError if the file cannot be built.
    #[wasm_bindgen(js_name = buildInto)]
    pub fn build_into(self, project: JsProject) -> Result<JsProjectFile, JsFileBuilderError> {
        let missing_name = self.name.is_none();
        let missing_protect_read = self.protect_read.is_none();
        let missing_protect_delete = self.protect_delete.is_none();
        let missing_bytes = self.bytes.is_none();

        let any_error =
            missing_name || missing_bytes || missing_protect_read || missing_protect_delete;

        let mut ws = wsw!();
        let project = ws.project_by_id_mut(project.uuid);
        if let Some(project) = project {
            if any_error {
                Err(JsFileBuilderError {
                    missing_name,
                    missing_bytes,
                    missing_protect_read,
                    missing_protect_delete,
                    project_not_found: false,
                })
            } else {
                let builder = FileBuilder {
                    name: self.name.clone().unwrap().into(),
                    protect_read: self.protect_read.unwrap(),
                    protect_delete: self.protect_delete.unwrap(),
                    bytes: self.bytes.unwrap(),
                };
                let protect = builder.protect();
                let uuid = project.add_file(builder.into());
                let js_file = JsProjectFile {
                    name: self.name.unwrap(),
                    uuid: uuid.into_js_array(),
                    project_uuid: Some(project.uuid()),
                    is_loaded: true,
                    can_load: false,
                    protect: JsProtect::from(protect),
                };
                Ok(js_file)
            }
        } else {
            Err(JsFileBuilderError {
                missing_name,
                missing_bytes,
                missing_protect_read,
                missing_protect_delete,
                project_not_found: true,
            })
        }
    }
}

#[wasm_bindgen(js_name = FileBuilderError)]
pub struct JsFileBuilderError {
    /// Whether the file name is missing.
    #[wasm_bindgen(readonly, js_name = missingName)]
    pub missing_name: bool,

    /// Whether the file contents are missing.
    #[wasm_bindgen(readonly, js_name = missingBytes)]
    pub missing_bytes: bool,

    /// Whether the protection level for reading the file is missing.
    #[wasm_bindgen(readonly, js_name = missingProtectRead)]
    pub missing_protect_read: bool,

    /// Whether the protection level for deleting the file is missing.
    #[wasm_bindgen(readonly, js_name = missingProtectDelete)]
    pub missing_protect_delete: bool,

    /// Whether the project was not found. This means that the project handle
    /// does not point to a valid project. This can happen when project was removed
    /// from the work session when the file builder was being set.
    #[wasm_bindgen(readonly, js_name = projectNotFound)]
    pub project_not_found: bool,
}

impl From<FileBuilder> for File {
    fn from(builder: FileBuilder) -> Self {
        Self {
            name: builder.name,
            bytes: Some(builder.bytes.to_vec()),
            can_load: true,
            protect: Protect::new(builder.protect_read, builder.protect_delete),
            uuid: Uuid::nil(),
        }
    }
}
