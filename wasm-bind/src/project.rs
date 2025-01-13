use base::table_data;
use js_sys::Uint8Array;
use smallvec::SmallVec;
use uuid::Uuid;

use serde_json::Value as JsonValue;

use crate::*;

pub struct Project {
    name: InterString,
    canvas: base::canvas::Canvas<JsonValue>,
    data: SmallVec<[table_data::Table; 1]>,
    files: SmallVec<[File; 1]>,
    uuid: Uuid,
}

impl Project {
    pub fn name(&self) -> &InterString {
        &self.name
    }

    pub fn data(&self) -> &[table_data::Table] {
        &self.data
    }

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

    pub fn uuid(&self) -> Uuid {
        self.uuid
    }

    pub fn canvas(&self) -> &base::canvas::Canvas<JsonValue> {
        &self.canvas
    }

    pub fn canvas_mut(&mut self) -> &mut base::canvas::Canvas<JsonValue> {
        &mut self.canvas
    }
}

/// A project handle that can be used to access project data from the current work session.
#[wasm_bindgen(js_name = Project)]
pub struct JsProject {
    uuid: Uuid,
}

#[wasm_bindgen(js_class = Project)]
impl JsProject {
    /// Get the files in the project. Note that some of the files can be declared
    /// but not actually loaded into the memory. This may be the case if the file is
    /// still being loaded or if the user has no permission to load the file content.
    /// However, this list will contain all files that the project has as long as
    /// the user has permission to view the project at least read-only.
    #[wasm_bindgen(getter, js_name = files)]
    pub fn get_files(&self) -> Vec<JsProjectFile> {
        let ws = work_session::work_session().read().unwrap();

        let project = if let Some(project) = ws.project_by_id(self.uuid) {
            project
        } else {
            // Project not found. Was removed from work session. This handle is invalid.
            // We return no files.
            return Vec::new();
        };

        project
            .files()
            .iter()
            .map(|file| JsProjectFile {
                name: file.name().to_js(),
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
        let ws = work_session::work_session().read().unwrap();
        let project = ws.project_by_id(self.uuid).unwrap();
        project.name().to_js()
    }

    /// Set the new name for the project.
    #[wasm_bindgen(setter, js_name = name)]
    pub fn set_name(&mut self, name: JsString) {
        let mut ws = work_session::work_session().write().unwrap();
        let project = ws.project_by_id_mut(self.uuid).unwrap();
        project.name.set_js(name);
    }

    /// Load the project with the given identifier, as returned by listing request.
    pub fn load(identifier: JsString) -> Result<JsProject, ProjectLoadError> {
        todo!()
    }

    /// Canvas of the project. This is where the nodes are placed.
    #[wasm_bindgen(getter)]
    pub fn canvas(&mut self) -> JsCanvas {
        JsCanvas {
            project_uuid: self.uuid,
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = ProjectLoadError)]
pub enum ProjectLoadError {
    NotFound,
    PermissionError,
    RequestError,
    ServerError,
}

/// Canvas of the project. This is where the nodes are placed.
#[derive(Debug, Copy, Clone)]
#[wasm_bindgen(js_name = Canvas)]
pub struct JsCanvas {
    project_uuid: Uuid,
}

#[wasm_bindgen(js_class = Canvas)]
impl JsCanvas {
    /// Add a new node to the canvas.
    #[wasm_bindgen(js_name = addNode)]
    pub fn add_node(&mut self, node: JsNodeStub) -> JsNode {
        todo!()
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
    pub fn node_iter(&self) -> JsNodeIter {
        todo!()
    }
}

/// Stub for a node. This is a configuration that allows to create a node in the canvas.
/// The same stub can be reused to effectively clone the node.
#[derive(Debug)]
#[wasm_bindgen(js_name = NodeStub)]
pub struct JsNodeStub {}

/// Node in the canvas. This is actual instance of the placed node in a project.
#[derive(Debug)]
#[wasm_bindgen(js_name = Node)]
pub struct JsNode {
    project_uuid: Uuid,
    node_id: base::canvas::Id,
}

#[wasm_bindgen(js_class = Node)]
impl JsNode {
    /// Get the stub that will allow to create a new node with the same configuration.
    #[wasm_bindgen(js_name = stub)]
    pub fn stub(&self) -> JsNodeStub {
        todo!()
    }

    /// Get the output pin at the given position.
    pub fn outAt(&self, position: u32) -> Option<JsNodePin> {
        todo!()
    }

    /// Get the input pin at the given position.
    pub fn inAt(&self, position: u32) -> Option<JsNodePin> {
        todo!()
    }

    /// Get the node identifier.
    pub fn id(&self) -> u32 {
        self.node_id.get()
    }

    /// Drop the node from the canvas. You should not use the node handle after this.
    pub fn drop(self) -> JsNodeStub {
        todo!()
    }
}

/// Node iterator to fetch nodes from the canvas. This exists so to avoid
/// duplicating big arrays of nodes in memory.
#[derive(Debug)]
#[wasm_bindgen(js_name = NodeIter)]
pub struct JsNodeIter {
    project_uuid: Uuid,
    pos: usize,
}

#[wasm_bindgen(js_class = NodeIter)]
impl JsNodeIter {
    /// Get the next node in the canvas.
    /// Returns `undefined` if there are no more nodes.
    pub fn next(&mut self) -> Option<JsNode> {
        todo!()
    }
}

/// Node pin. This is a connection point on a node.
#[derive(Debug)]
#[wasm_bindgen(js_name = NodePin)]
pub struct JsNodePin {
    project_uuid: Uuid,
    node_id: base::canvas::Id,
    ordinal: u32,
    is_output: bool,
}

#[wasm_bindgen(js_class = NodePin)]
impl JsNodePin {
    /// Get the data type of the pin.
    /// If the pin has unknown data type, this will return `undefined`.
    /// If the pin has nested data type, this will return an array of data types, where
    /// the first element is the outermost data type.
    /// Normal data type will return a single element array.
    #[wasm_bindgen(getter, js_name = dataType)]
    pub fn data_type(&self) -> Option<Vec<JsDataType>> {
        todo!()
    }

    /// Whether the pin is an output pin (`true`). If it is an input pin, this returns `false`.
    #[wasm_bindgen(getter, js_name = isOutput)]
    pub fn is_output(&self) -> bool {
        self.is_output
    }

    /// Whether the pin accepts optional values.
    #[wasm_bindgen(getter, js_name = isNullable)]
    pub fn is_nullable(&self) -> bool {
        todo!()
    }

    /// Get the position of the pin on the node.
    #[wasm_bindgen(getter)]
    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    /// Get the node that this pin is connected to. `undefined` if not connected.
    #[wasm_bindgen(js_name = connectedTo)]
    pub fn connected_to(&self) -> Option<JsNodePin> {
        todo!()
    }

    /// Peek into the data that this pin receiver or transmits.
    /// This may be useful for UI to show the data for user debugging or analysis.
    #[wasm_bindgen(js_name = peekInto)]
    pub fn peek_into(&self) -> JsPeekFlow {
        todo!()
    }

    /// Get the errors that this pin is associated with.
    pub fn errors(&self) -> Vec<JsNodePinError> {
        todo!()
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
}

#[derive(Debug)]
#[wasm_bindgen(js_name = DataType)]
pub struct JsDataType {
    /// The kind of the data type.
    #[wasm_bindgen(readonly)]
    pub kind: JsDataTypeKind,

    /// Whether this data type is optional or required.
    #[wasm_bindgen(readonly, js_name = isNullable)]
    pub is_nullable: bool,
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
    /// If the data type is nested, this will return an array of data types, where
    /// the first element is the outermost data type.
    #[wasm_bindgen(js_name = dataType)]
    pub fn data_type(&self) -> Vec<JsDataType> {
        todo!()
    }

    /// Shuffle and get the array of unique values, up to the given limit.
    /// Array will have less elements if the flow does not have enough unique values.
    #[wasm_bindgen(js_name = shuffled)]
    pub fn shuffled(limit: usize) -> Vec<JsFlowValue> {
        todo!()
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

pub struct File {
    name: InterString,

    /// The file's contents. This is `None` if the file is not loaded into memory.
    bytes: Option<Vec<u8>>,

    /// Whether the file can be loaded into memory.
    can_load: bool,

    protect: Protect,

    uuid: Uuid,
}

impl File {
    pub fn name(&self) -> &InterString {
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
    pub async fn load(&self) -> Result<(), FileLoadError> {
        todo!()
    }

    /// Remove the file from the associated project.
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
    name: InterString,
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
    #[wasm_bindgen(js_name = buildInto)]
    pub fn build_into(self, project: JsProject) -> Result<JsProjectFile, JsFileBuilderError> {
        let missing_name = self.name.is_none();
        let missing_protect_read = self.protect_read.is_none();
        let missing_protect_delete = self.protect_delete.is_none();
        let missing_bytes = self.bytes.is_none();

        let any_error =
            missing_name || missing_bytes || missing_protect_read || missing_protect_delete;

        let mut ws = work_session::work_session().write().unwrap();
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
