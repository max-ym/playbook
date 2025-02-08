use std::collections::HashMap;

use base::table_data;
use chrono::Datelike;
use js_sys::{JsString, RegExp, Uint8Array};
use log::error;
use smallvec::SmallVec;
use uuid::Uuid;

use serde_json::Value as JsonValue;
use wasm_bindgen::prelude::*;

use crate::{MyUuid, PermissionError};

macro_rules! wsr {
    () => {
        crate::work_session::work_session().read().unwrap()
    };
}
macro_rules! wsw {
    () => {
        crate::work_session::work_session().write().unwrap()
    };
}

pub struct Project {
    name: JsString,
    canvas: base::canvas::Canvas<JsonValue>,
    data: SmallVec<[table_data::Table; 1]>,
    files: SmallVec<[File; 1]>,
    uuid: Uuid,
}

impl Project {
    pub fn name(&self) -> &JsString {
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

    pub(crate) fn canvas(&self) -> &base::canvas::Canvas<JsonValue> {
        &self.canvas
    }

    pub(crate) fn canvas_mut(&mut self) -> &mut base::canvas::Canvas<JsonValue> {
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
            error!("Project not found. Was removed from work session. Project handle is invalid.");
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
            error!("Project not found. Was removed from work session. Project handle is invalid.");
        }
    }

    /// Load the project with the given identifier, as returned by listing request.
    ///
    /// This will throw ProjectLoadError if the project cannot be loaded.
    ///
    /// # Fake Server
    /// Returns a project handle for this identifier if such project exists
    /// in the work session already.
    pub fn load(identifier: JsString) -> Result<JsProject, ProjectLoadError> {
        if cfg!(feature = "fake_server") {
            let identifier = String::from(identifier);
            let mut ws = wsw!();

            let uuid = Uuid::parse_str(&identifier).map_err(|_| ProjectLoadError::NotFound)?;
            let project = ws.project_by_id_mut(uuid);
            if let Some(project) = project {
                Ok(JsProject {
                    uuid: project.uuid(),
                })
            } else {
                error!("Failed to load project by identifier `{identifier}`");
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
    pub fn add_node(&mut self, node: JsNodeStub, meta: JsValue) -> Result<JsNode, JsError> {
        let mut ws = wsw!();
        let project = ws.project_by_id_mut(self.project_uuid);
        if let Some(project) = project {
            let meta =
                serde_wasm_bindgen::from_value(meta).map_err(|e| JsError::new(&e.to_string()))?;
            let node = project.canvas_mut().add_node(node.stub.into(), meta);
            Ok(JsNode {
                project_uuid: self.project_uuid,
                node_id: node,
            })
        } else {
            error!("Project not found. Was removed from work session. Project handle is invalid.");
            Err(JsError::new("Project not found, handle is invalid"))
        }
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
        JsNodeIter {
            project_uuid: self.project_uuid,
            pos: 0,
        }
    }
}

/// Stub for a node. This is a configuration that allows to create a node in the canvas.
/// The same stub can be reused to effectively clone the node.
#[derive(Debug, Clone)]
#[wasm_bindgen(js_name = NodeStub)]
pub struct JsNodeStub {
    stub: base::canvas::NodeStub,
}

impl From<base::canvas::NodeStub> for JsNodeStub {
    fn from(stub: base::canvas::NodeStub) -> Self {
        Self { stub }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = FileNodeStub)]
pub struct JsFileNodeStub {}

#[wasm_bindgen(js_class = FileNodeStub)]
impl JsFileNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsFileNodeStub {
        JsFileNodeStub {}
    }

    #[wasm_bindgen(getter)]
    pub fn stub(&self) -> JsNodeStub {
        base::canvas::NodeStub::File.into()
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = SplitByNodeStub)]
pub struct JsSplitByNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub regex: RegExp,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = SplitByNodeStub)]
impl JsSplitByNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn from_regex(js_regex: RegExp) -> JsSplitByNodeStub {
        use lazy_regex::regex::Regex;
        let s = String::from(js_regex.source());
        let regex = Regex::new(&s).expect("it was valid for JS and should be for us as well");

        JsSplitByNodeStub {
            regex: js_regex,
            stub: base::canvas::NodeStub::SplitBy { regex }.into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = InputNodeStub)]
pub struct JsInputNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone, js_name = validNames)]
    pub valid_names: Vec<JsString>,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = InputNodeStub)]
impl JsInputNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(js_valid_names: Vec<JsString>) -> JsInputNodeStub {
        let valid_names = js_valid_names
            .iter()
            .map(|s| String::from(s).into())
            .collect();
        let stub = base::canvas::NodeStub::Input { valid_names }.into();
        JsInputNodeStub {
            valid_names: js_valid_names,
            stub,
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = DropNodeStub)]
pub struct JsDropNodeStub {}

#[wasm_bindgen(js_class = DropNodeStub)]
impl JsDropNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsDropNodeStub {
        JsDropNodeStub {}
    }

    #[wasm_bindgen(getter)]
    pub fn stub(&self) -> JsNodeStub {
        base::canvas::NodeStub::Drop.into()
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = OutputNodeStub)]
pub struct JsOutputNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub ident: JsString,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = OutputNodeStub)]
impl JsOutputNodeStub {
    /// Create a new output node stub.
    ///
    /// # Errors
    /// Passed in string should be valid identifier:
    /// - It should start with a letter or underscore.
    /// - It should contain only letters, digits, and underscores.
    /// - No longer than 64 characters.
    #[wasm_bindgen(constructor)]
    pub fn new(name: JsString) -> Result<JsOutputNodeStub, JsError> {
        let s = String::from(name.clone());
        if s.len() > 64 {
            Err(JsError::new("Name is too long"))
        } else if lazy_regex::regex_is_match!(r"^[a-zA-Z_][a-zA-Z0-9_]*$", &s) {
            let stub = base::canvas::NodeStub::Output { ident: s.into() }.into();
            Ok(JsOutputNodeStub { ident: name, stub })
        } else {
            Err(JsError::new("Invalid name"))
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = LowercaseNodeStub)]
pub struct JsLowercaseNodeStub {}

#[wasm_bindgen(js_class = LowercaseNodeStub)]
impl JsLowercaseNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsLowercaseNodeStub {
        JsLowercaseNodeStub {}
    }

    #[wasm_bindgen(getter)]
    pub fn stub(&self) -> JsNodeStub {
        use base::canvas::*;
        NodeStub::StrOp(StrOp::Lowercase).into()
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = UppercaseNodeStub)]
pub struct JsUppercaseNodeStub {}

#[wasm_bindgen(js_class = UppercaseNodeStub)]
impl JsUppercaseNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsUppercaseNodeStub {
        JsUppercaseNodeStub {}
    }

    #[wasm_bindgen(getter)]
    pub fn stub(&self) -> JsNodeStub {
        use base::canvas::*;
        NodeStub::StrOp(StrOp::Uppercase).into()
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = StripNodeStub)]
pub struct JsStripNodeStub {
    #[wasm_bindgen(readonly)]
    pub trim_whitespace: bool,
    #[wasm_bindgen(readonly)]
    pub trim_end_whitespace: bool,
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub remove: Vec<JsString>,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = StripNodeStub)]
impl JsStripNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(
        trim_whitespace: bool,
        trim_end_whitespace: bool,
        remove: Vec<JsString>,
    ) -> JsStripNodeStub {
        use base::canvas::*;

        let stub = NodeStub::StrOp(StrOp::Strip {
            trim_whitespace,
            trim_end_whitespace,
            remove: remove.iter().map(|s| String::from(s).into()).collect(),
        })
        .into();

        JsStripNodeStub {
            trim_whitespace,
            trim_end_whitespace,
            remove,
            stub,
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = CompareNodeStub)]
pub struct JsCompareNodeStub {
    #[wasm_bindgen(readonly)]
    pub eq: bool,
}

#[wasm_bindgen(js_class = CompareNodeStub)]
impl JsCompareNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn eq() -> JsCompareNodeStub {
        JsCompareNodeStub { eq: true }
    }

    #[wasm_bindgen(constructor)]
    pub fn ne() -> JsCompareNodeStub {
        JsCompareNodeStub { eq: false }
    }

    #[wasm_bindgen(getter)]
    pub fn stub(&self) -> JsNodeStub {
        base::canvas::NodeStub::Compare { eq: self.eq }.into()
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = OrderingNodeStub)]
pub struct JsOrderingNodeStub;

#[wasm_bindgen(js_class = OrderingNodeStub)]
impl JsOrderingNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn ordering() -> JsOrderingNodeStub {
        JsOrderingNodeStub
    }

    #[wasm_bindgen(getter)]
    pub fn stub(&self) -> JsNodeStub {
        base::canvas::NodeStub::Ordering.into()
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = RegexNodeStub)]
pub struct JsRegexNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub regex: RegExp,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = RegexNodeStub)]
impl JsRegexNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn from_regex(js_regex: RegExp) -> JsRegexNodeStub {
        use lazy_regex::regex::Regex;
        let s = String::from(js_regex.source());
        let regex = Regex::new(&s).expect("JS thinks it is valid...");

        JsRegexNodeStub {
            regex: js_regex,
            stub: base::canvas::NodeStub::Regex(regex).into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = MapNodeStubBuilder)]
pub struct JsMapNodeStubBuilder {
    /// Map given pattern to given value(s). Many values can be given,
    /// if the node has several outputs, corresponding to each value position.
    map: HashMap<base::canvas::Pat, base::canvas::Value>,

    #[wasm_bindgen(getter_with_clone)]
    pub wildcard: Vec<JsDataInstance>,
}

#[wasm_bindgen(js_class = MapNodeStubBuilder)]
impl JsMapNodeStubBuilder {
    /// Add "or pattern" to the map. This is a pattern that matches any of the given keys.
    /// The output value will be set to the given value.
    /// If this function was already called for given keys, the value will remain as was set
    /// by the first call.
    pub fn or_pat(&mut self, keys: Vec<JsDataInstance>, value: JsDataInstance) {
        let pat = base::canvas::Pat::from_variants(keys.into_iter().map(|v| v.value).collect());
        self.map.entry(pat).or_insert(value.value);
    }

    /// Validate and build the map node stub.
    pub fn build(self) -> Result<JsMapNodeStub, JsError> {
        let wildcard = if self.wildcard.is_empty() {
            None
        } else if self.wildcard.len() == 1 {
            let w = self.wildcard.into_iter().next().unwrap();
            Some(w.value)
        } else {
            Some(base::canvas::Value::Array(
                self.wildcard.into_iter().map(|v| v.value).collect(),
            ))
        };

        if self.map.is_empty() {
            return Ok(JsMapNodeStub {
                stub: base::canvas::NodeStub::Map {
                    tuples: Default::default(),
                    wildcard,
                }
                .into(),
            });
        }

        // If keys/values are arrays, all corresponding arrays should be the same size.
        // All values and keys should have the same type.
        let (first_pat, first_val) = self
            .map
            .iter()
            .next()
            .expect("at least one key should exist, per guard above");

        for (pat, val) in self.map.iter().skip(1) {
            if !pat.is_compatible_with(first_pat) {
                return Err(JsError::new("all keys should have the same types"));
            }
            if !val.is_same_type(first_val) {
                return Err(JsError::new("all values should have the same type"));
            }
        }

        Ok(JsMapNodeStub {
            stub: base::canvas::NodeStub::Map {
                tuples: self.map.into_iter().collect(),
                wildcard,
            }
            .into(),
        })
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = MapNodeStub)]
pub struct JsMapNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = MapNodeStub)]
impl JsMapNodeStub {
    #[wasm_bindgen(getter, js_name = map)]
    pub fn map(&self) -> js_sys::Map {
        if let base::canvas::NodeStub::Map { tuples, .. } = &self.stub.stub {
            let map = js_sys::Map::new();

            for (pat, value) in tuples.clone() {
                map.set(
                    &JsDataInstance {
                        value: pat.into_value(),
                    }
                    .into(),
                    &JsDataInstance { value }.into(),
                );
            }

            map
        } else {
            unreachable!("stub is not Map, but object is MapNodeStub");
        }
    }
}

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
    ///
    /// # Errors
    /// If the project or the node is no longer valid, this will return an error.
    #[wasm_bindgen(js_name = stub)]
    pub fn stub(&self) -> Result<JsNodeStub, JsError> {
        let ws = wsr!();
        let maybe_project = ws.project_by_id(self.project_uuid);
        if let Some(project) = maybe_project {
            if let Some(node) = project.canvas().node(self.node_id) {
                Ok(node.stub.clone().into())
            } else {
                Err(JsError::new("Node not found"))
            }
        } else {
            Err(JsError::new("Project not found"))
        }
    }

    /// Get the output pin at the given position.
    #[wasm_bindgen(js_name = outAt)]
    pub fn out_at(&self, position: u32) -> Result<JsNodePin, JsError> {
        let ws = wsr!();
        let project = ws
            .project_by_id(self.project_uuid)
            .ok_or_else(|| JsError::new("Project not found"))?;
        let node = project
            .canvas()
            .node(self.node_id)
            .ok_or_else(|| JsError::new("Node not found"))?;
        if let Some(ordinal) = node.stub.real_output_pin_idx(position as _) {
            Ok(JsNodePin {
                project_uuid: self.project_uuid,
                node_id: self.node_id,
                ordinal: ordinal as _,
            })
        } else {
            Err(JsError::new("Output pin not found"))
        }
    }

    /// Get the input pin at the given position.
    #[wasm_bindgen(js_name = inAt)]
    pub fn in_at(&self, position: u32) -> Option<JsNodePin> {
        let ws = wsr!();
        let project = ws.project_by_id(self.project_uuid)?;
        let node = project.canvas().node(self.node_id)?;
        let ordinal = node.stub.real_input_pin_idx(position as _)?;
        Some(JsNodePin {
            project_uuid: self.project_uuid,
            node_id: self.node_id,
            ordinal,
        })
    }

    /// Get the node identifier.
    pub fn id(&self) -> u32 {
        self.node_id.get()
    }

    /// Drop the node from the canvas. You should not use the node handle after this.
    /// 
    /// This also returns the stub of the node, so you can recreate it later.
    pub fn drop(self) -> Result<JsNodeStub, JsError> {
        let stub = self.stub()?;

        let mut ws = wsw!();
        let project = ws.project_by_id_mut(self.project_uuid);
        if let Some(project) = project {
            project.canvas_mut().remove_node(self.node_id);
            Ok(stub)
        } else {
            error!("Project not found. Was removed from work session. Project handle is invalid.");
            Err(JsError::new("Project not found, handle is invalid"))
        }
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
    ordinal: base::canvas::PinOrder,
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
        todo!()
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
    #[wasm_bindgen(getter)]
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
    Option,
}

#[derive(Debug, Clone)]
#[wasm_bindgen(js_name = DataType)]
pub struct JsDataType {
    repr: base::canvas::PrimitiveType,
}

#[wasm_bindgen(js_class = DataType)]
impl JsDataType {
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> JsDataTypeKind {
        use base::canvas::PrimitiveType::*;
        use JsDataTypeKind as J;
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
            Predicate(_) => J::Predicate,
            Result(_) => J::Result,
            Option(_) => J::Option,
        }
    }

    /// Inner data type if this is a nested data type.
    #[wasm_bindgen(getter)]
    pub fn inner(&self) -> std::option::Option<JsDataType> {
        use base::canvas::PrimitiveType::*;
        use std::ops::Deref;
        match &self.repr {
            Array(inner) => Some(inner.deref().to_owned().into()),
            Result(inner) => Some(inner.deref().to_owned().into()),
            Option(inner) => Some(inner.deref().to_owned().into()),

            Predicate(_) => None,

            Int => None,
            Uint => None,
            Unit => None,
            Moneraty => None,
            Date => None,
            DateTime => None,
            Time => None,
            Bool => None,
            Str => None,
            Ordering => None,
            File => None,
            Record => None,
        }
    }
}

impl From<base::canvas::PrimitiveType> for JsDataType {
    fn from(repr: base::canvas::PrimitiveType) -> Self {
        Self { repr }
    }
}

impl From<JsDataType> for base::canvas::PrimitiveType {
    fn from(js: JsDataType) -> Self {
        js.repr
    }
}

/// A value as an instance of a data type supported by the canvas.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
#[wasm_bindgen(js_name = Value)]
pub struct JsDataInstance {
    value: base::canvas::Value,
}

#[wasm_bindgen(js_class = Value)]
impl JsDataInstance {
    /// Get the data type of the value.
    #[wasm_bindgen(getter, js_name = dataType)]
    pub fn data_type(&self) -> JsDataType {
        self.value.type_of().into()
    }

    /// Get the value as a JS value.
    #[wasm_bindgen(getter)]
    pub fn value(&self) -> JsValue {
        use base::canvas::Value::*;
        use bigdecimal::ToPrimitive;
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
            Predicate(predicate) => JsPredicate {
                inputs: predicate.inputs.iter().map(|t| t.clone().into()).collect(),
                outputs: predicate.outputs.iter().map(|t| t.clone().into()).collect(),
            }
            .into(),
            Result { value, is_ok } => JsResult {
                value: JsDataInstance {
                    value: value.deref().clone(),
                }
                .into(),
                is_ok: *is_ok,
            }
            .into(),
            Option { value, is_some } => JsOption {
                value: JsDataInstance {
                    value: value.deref().clone(),
                }
                .into(),
                is_some: *is_some,
            }
            .into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = Time)]
pub struct JsTime {
    pub hour: i8,
    pub minute: i8,
    pub second: i8,
}

#[derive(Debug)]
#[wasm_bindgen(js_name = Monetary)]
pub struct JsMonetary {
    pub amount: f64,
}

#[derive(Debug)]
#[wasm_bindgen(js_name = Ordering)]
pub enum JsOrdering {
    Less,
    Equal,
    Greater,
}

#[derive(Debug)]
#[wasm_bindgen(js_name = Result)]
pub struct JsResult {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub value: JsDataInstance,

    #[wasm_bindgen(readonly, js_name = isOk)]
    pub is_ok: bool,
}

#[derive(Debug)]
#[wasm_bindgen(js_name = Option)]
pub struct JsOption {
    /// The value is always present, even on "None" variant.
    /// It is not a valid data instance if `isSome` is `false`,
    /// however this value still can be used to infer the data type, as it is
    /// guaranteed to be the same as for the value of the `Some` variant.
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub value: JsDataInstance,

    /// Whether the value is present. If this is `false`, the `value` is not a valid data instance.
    #[wasm_bindgen(readonly, js_name = isSome)]
    pub is_some: bool,
}

#[derive(Debug)]
#[wasm_bindgen(js_name = Predicate)]
pub struct JsPredicate {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub inputs: Vec<JsDataType>,

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
    /// If the data type is nested, this will return an array of data types, where
    /// the first element is the outermost data type.
    #[wasm_bindgen(js_name = dataType)]
    pub fn data_type(&self) -> Vec<JsDataType> {
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
