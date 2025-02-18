use std::collections::HashMap;

use base::{canvas, table_data};
use chrono::Datelike;
use js_sys::{JsString, RegExp, Uint8Array};
use log::error;
use smallvec::SmallVec;
use thiserror::Error;
use uuid::Uuid;

use serde_json::Value as JsonValue;
use wasm_bindgen::prelude::*;

use crate::{
    work_session::{self, wsr, wsw, DetectChangedStack},
    InvalidHandleError, MyUuid, PermissionError,
};

pub struct Project {
    name: JsString,
    canvas: canvas::Canvas<JsonValue>,
    data: SmallVec<[table_data::Table; 1]>,
    files: SmallVec<[File; 1]>,
    uuid: Uuid,
}

impl Project {
    pub fn zero() -> Self {
        Self {
            name: JsString::from("Unnamed"),
            canvas: canvas::Canvas::new(),
            data: SmallVec::new(),
            files: SmallVec::new(),
            uuid: Uuid::nil(),
        }
    }

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

    pub(crate) fn canvas(&self) -> &canvas::Canvas<JsonValue> {
        &self.canvas
    }

    pub(crate) fn canvas_mut(&mut self) -> &mut canvas::Canvas<JsonValue> {
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
    /// # Errors
    /// This will throw ProjectLoadError if the project cannot be loaded, or already exists
    /// in the current work session.
    ///
    /// # Fake Server
    /// UUID 0 allows to load empty project as if it was loaded from the server.
    pub fn load(identifier: JsString) -> Result<JsProject, ProjectLoadError> {
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

#[wasm_bindgen(js_class = Canvas)]
impl JsCanvas {
    /// Add a new node to the canvas.
    #[wasm_bindgen(js_name = addNode)]
    pub fn add_node(&mut self, stub: JsNodeStub, meta: JsValue) -> Result<JsNode, JsError> {
        let mut ws = wsw!();
        let project = ws.project_by_id_mut(self.project_uuid);
        if let Some(project) = project {
            let node_id = project.add_node(
                stub.stub,
                serde_wasm_bindgen::from_value(meta).map_err(|e| {
                    error!("Failed to deserialize meta data for the node. {:?}", e);
                    JsError::new("Failed to deserialize meta data for the node")
                })?,
            );
            Ok(JsNode {
                project_uuid: self.project_uuid,
                node_id,
            })
        } else {
            error!("Project not found. Was removed from work session. Project handle is invalid.");
            Err(InvalidHandleError.into())
        }
    }

    /// Add a new edge between two pins.
    #[wasm_bindgen(js_name = addEdge)]
    pub fn add_edge(&mut self, from: JsNodePin, to: JsNodePin) -> Result<(), JsError> {
        use canvas::{Edge, InputPin, OutputPin, Pin};

        let mut ws = wsw!();
        let project = ws.project_by_id_mut(self.project_uuid);
        if let Some(project) = project {
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
        } else {
            error!("Project not found. Was removed from work session. Project handle is invalid.");
            Err(InvalidHandleError.into())
        }
    }

    /// Remove an edge between two pins.
    #[wasm_bindgen(js_name = removeEdge)]
    pub fn remove_edge(&mut self, from: JsNodePin, to: JsNodePin) -> Result<(), JsError> {
        use canvas::{Edge, InputPin, OutputPin, Pin};

        let mut ws = wsw!();
        let project = ws.project_by_id_mut(self.project_uuid);
        if let Some(project) = project {
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
        } else {
            error!("Project not found. Was removed from work session. Project handle is invalid.");
            Err(InvalidHandleError.into())
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
    pub fn node_iter(&self) -> Result<JsNodeIter, JsError> {
        let ws = wsr!();
        let project = ws
            .project_by_id(self.project_uuid)
            .ok_or(InvalidHandleError)?;

        Ok(JsNodeIter {
            detect_change: project.detect_changed_stack(),
            project_uuid: self.project_uuid,
            pos: 0,
        })
    }
}

/// Stub for a node. This is a configuration that allows to create a node in the canvas.
/// The same stub can be reused to effectively clone the node.
#[derive(Debug, Clone)]
#[wasm_bindgen(js_name = NodeStub)]
pub struct JsNodeStub {
    stub: canvas::NodeStub,
}

#[wasm_bindgen(js_class = NodeStub)]
impl JsNodeStub {
    /// Get kind of the node stub.
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> JsNodeStubKind {
        use canvas::NodeStub as R;
        use JsNodeStubKind as JS;

        match &self.stub {
            R::File => JS::File,
            R::SplitBy { .. } => JS::SplitBy,
            R::Input { .. } => JS::Input,
            R::Drop => JS::Drop,
            R::Output { .. } => JS::Output,
            R::StrOp(op) => match op {
                canvas::StrOp::Lowercase => JS::Lowercase,
                canvas::StrOp::Uppercase => JS::Uppercase,
                canvas::StrOp::Strip { .. } => JS::Strip,
            },
            R::Compare { .. } => JS::Compare,
            R::Ordering => JS::Ordering,
            R::Regex(_) => JS::Regex,
            R::FindRecord { .. } => JS::FindRecord,
            R::Map { .. } => JS::Map,
            R::List { .. } => JS::List,
            R::IfElse { .. } => JS::IfElse,
            R::OkOrErr => JS::OkOrErr,
            R::Validate { .. } => JS::Validate,
            R::SelectFirst { .. } => JS::SelectFirst,
            R::ExpectOne { .. } => JS::ExpectOne,
            R::ExpectSome { .. } => JS::ExpectSome,
            R::Match { .. } => JS::Match,
            R::Func { .. } => JS::Func,
            R::ParseDateTime { .. } => JS::ParseDateTime,
            R::ParseMonetary { .. } => JS::ParseMonetary,
            R::ParseInt { .. } => JS::ParseInt,
            R::Constant(_) => JS::Constant,
            R::OkOrCrash { .. } => JS::OkOrCrash,
            R::Crash { .. } => JS::Crash,
            R::Unreachable { .. } => JS::Unreachable,
            R::Todo { .. } => JS::Todo,
            R::Comment { .. } => JS::Comment,
        }
    }
}

impl From<canvas::NodeStub> for JsNodeStub {
    fn from(stub: canvas::NodeStub) -> Self {
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
        canvas::NodeStub::File.into()
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
            stub: canvas::NodeStub::SplitBy { regex }.into(),
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
        let stub = canvas::NodeStub::Input { valid_names }.into();
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
        canvas::NodeStub::Drop.into()
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
            let stub = canvas::NodeStub::Output { ident: s.into() }.into();
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
        use canvas::*;
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
        use canvas::*;
        NodeStub::StrOp(StrOp::Uppercase).into()
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = StripNodeStub)]
pub struct JsStripNodeStub {
    #[wasm_bindgen(readonly, js_name = trimWhitespace)]
    pub trim_whitespace: bool,
    #[wasm_bindgen(readonly, js_name = trimEndWhitespace)]
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
        use canvas::*;

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
    pub fn eq() -> JsCompareNodeStub {
        JsCompareNodeStub { eq: true }
    }

    pub fn ne() -> JsCompareNodeStub {
        JsCompareNodeStub { eq: false }
    }

    #[wasm_bindgen(getter)]
    pub fn stub(&self) -> JsNodeStub {
        canvas::NodeStub::Compare { eq: self.eq }.into()
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
        canvas::NodeStub::Ordering.into()
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
        let regex = js_regex.into_rust();
        JsRegexNodeStub {
            regex: js_regex,
            stub: canvas::NodeStub::Regex(regex).into(),
        }
    }
}

trait IntoRegex {
    fn into_rust(&self) -> lazy_regex::regex::Regex;
}

impl IntoRegex for RegExp {
    fn into_rust(&self) -> lazy_regex::regex::Regex {
        use lazy_regex::regex::Regex;
        let s = String::from(self.source());
        Regex::new(&s).expect("JS thinks it is valid...")
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = MapNodeStubBuilder)]
pub struct JsMapNodeStubBuilder {
    /// Map given pattern to given value(s). Many values can be given,
    /// if the node has several outputs, corresponding to each value position.
    map: HashMap<canvas::Pat, canvas::Value>,

    #[wasm_bindgen(getter_with_clone)]
    pub wildcard: Vec<JsDataInstance>,
}

#[wasm_bindgen(js_class = MapNodeStubBuilder)]
impl JsMapNodeStubBuilder {
    /// Add "or pattern" to the map. This is a pattern that matches any of the given keys.
    /// The output value will be set to the given value.
    /// If this function was already called for given keys, the value will remain as was set
    /// by the first call.
    #[wasm_bindgen(js_name = orPat)]
    pub fn or_pat(&mut self, keys: Vec<JsDataInstance>, value: JsDataInstance) {
        let pat = canvas::Pat::from_variants(keys.into_iter().map(|v| v.value).collect());
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
            Some(canvas::Value::Array(
                self.wildcard.into_iter().map(|v| v.value).collect(),
            ))
        };

        if self.map.is_empty() {
            return Ok(JsMapNodeStub {
                stub: canvas::NodeStub::Map {
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
            stub: canvas::NodeStub::Map {
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
        if let canvas::NodeStub::Map { tuples, .. } = &self.stub.stub {
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

#[derive(Debug)]
#[wasm_bindgen(js_name = ListNodeStub)]
pub struct JsListNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen]
impl JsListNodeStub {
    /// Add element to the list node stub.
    /// This will check if the element is of the same type as the first element in the list.
    /// If the list is empty, the element will be added.
    ///
    /// # Errors
    /// If the element is not of the same type as the first element, this will throw an error.
    #[wasm_bindgen(js_name = addElement)]
    pub fn add_element(&mut self, element: JsDataInstance) -> Result<(), JsInvalidTypeError> {
        if let canvas::NodeStub::List { values } = &mut self.stub.stub {
            if let Some(first) = values.first() {
                if !element.value.is_same_type(first) {
                    return Err(JsInvalidTypeError {
                        got: element.value.type_of().into(),
                        expected: first.type_of().into(),
                    });
                }
            }

            values.push(element.value);
            Ok(())
        } else {
            unreachable!("stub is not List, but object is ListNodeStub");
        }
    }

    /// Get the elements in the list node stub.
    #[wasm_bindgen(getter, js_name = elements)]
    pub fn elements(&self) -> Vec<JsDataInstance> {
        if let canvas::NodeStub::List { values } = &self.stub.stub {
            values
                .iter()
                .map(|v| JsDataInstance { value: v.clone() })
                .collect()
        } else {
            unreachable!("stub is not List, but object is ListNodeStub");
        }
    }

    /// Remove element at the given index. If there is no element at the given index,
    /// this will do nothing.
    #[wasm_bindgen(js_name = removeElementAt)]
    pub fn remove_element_at(&mut self, idx: usize) {
        if let canvas::NodeStub::List { values } = &mut self.stub.stub {
            if idx < values.len() {
                let _ = values.remove(idx);
            }
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = InvalidTypeError)]
pub struct JsInvalidTypeError {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub got: JsDataType,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub expected: JsDataType,
}

#[derive(Debug)]
#[wasm_bindgen(js_name = IfElseNodeStub)]
pub struct JsIfElseNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = IfElseNodeStub)]
impl JsIfElseNodeStub {
    /// Create a new if-else node stub with given amount of data inputs. Note one more first
    /// input pin that is added on top to accept boolean value for the evaluated condition.
    #[wasm_bindgen(constructor)]
    pub fn new(inputs: usize) -> JsIfElseNodeStub {
        JsIfElseNodeStub {
            stub: canvas::NodeStub::IfElse {
                inputs: inputs as canvas::PinOrder,
            }
            .into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = OkOrErrNodeStub)]
pub struct JsOkOrErrNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = OkOrErrNodeStub)]
impl JsOkOrErrNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsOkOrErrNodeStub {
        JsOkOrErrNodeStub {
            stub: canvas::NodeStub::OkOrErr.into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = ValidateNodeStub)]
pub struct JsValidateNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = ValidateNodeStub)]
impl JsValidateNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsValidateNodeStub {
        todo!("ValidateNodeStub")
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = SelectFirstNodeStub)]
pub struct JsSelectFirstNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = SelectFirstNodeStub)]
impl JsSelectFirstNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(inputs: usize) -> JsSelectFirstNodeStub {
        JsSelectFirstNodeStub {
            stub: canvas::NodeStub::SelectFirst {
                inputs: inputs as canvas::PinOrder,
            }
            .into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = ExpectOneNodeStub)]
pub struct JsExpectOneNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = ExpectOne)]
impl JsExpectOneNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(inputs: usize) -> JsExpectOneNodeStub {
        JsExpectOneNodeStub {
            stub: canvas::NodeStub::ExpectOne {
                inputs: inputs as canvas::PinOrder,
            }
            .into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = ExpectSomeNodeStub)]
pub struct JsExpectSomeNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = ExpectSomeNodeStub)]
impl JsExpectSomeNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(msg: String) -> JsExpectSomeNodeStub {
        JsExpectSomeNodeStub {
            stub: canvas::NodeStub::ExpectSome { msg: msg.into() }.into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = MatchNodeStub)]
pub struct JsMatchNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = MatchNode)]
impl JsMatchNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsMatchNodeStub {
        todo!("Similar as Map, DRY code?")
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = FuncNodeStub)]
pub struct JsFuncNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = FuncNodeStub)]
impl JsFuncNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsFuncNodeStub {
        todo!("predicates and functions?")
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = ParseDateTimeNodeStub)]
pub struct JsParseDateTimeNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub format: JsString,

    #[wasm_bindgen(readonly)]
    pub date: bool,

    #[wasm_bindgen(readonly)]
    pub time: bool,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = ParseDateTimeNodeStub)]
impl JsParseDateTimeNodeStub {
    fn new(format: JsString, date: bool, time: bool) -> Self {
        let rust_format = String::from(&format).into();
        JsParseDateTimeNodeStub {
            format,
            date,
            time,
            stub: canvas::NodeStub::ParseDateTime {
                format: rust_format,
                date,
                time,
            }
            .into(),
        }
    }

    #[wasm_bindgen(js_name = date)]
    pub fn date(format: JsString) -> JsParseDateTimeNodeStub {
        Self::new(format, true, false)
    }

    #[wasm_bindgen(js_name = time)]
    pub fn time(format: JsString) -> JsParseDateTimeNodeStub {
        Self::new(format, false, true)
    }

    #[wasm_bindgen(js_name = dateTime)]
    pub fn date_time(format: JsString) -> JsParseDateTimeNodeStub {
        Self::new(format, true, true)
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = ParseMonetaryNodeStub)]
pub struct JsParseMonetaryNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub regex: RegExp,
}

#[wasm_bindgen(js_class = ParseMonetaryNodeStub)]
impl JsParseMonetaryNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(js_regex: RegExp) -> JsParseMonetaryNodeStub {
        let regex = js_regex.into_rust();
        JsParseMonetaryNodeStub {
            regex: js_regex,
            stub: canvas::NodeStub::ParseMonetary { regex }.into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = ParseIntNodeStub)]
pub struct JsParseIntNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,

    #[wasm_bindgen(readonly, js_name = isSigned)]
    pub is_signed: bool,
}

#[wasm_bindgen(js_class = ParseIntNodeStub)]
impl JsParseIntNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(is_signed: bool) -> JsParseIntNodeStub {
        JsParseIntNodeStub {
            is_signed,
            stub: canvas::NodeStub::ParseInt { signed: is_signed }.into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = ConstantNodeStub)]
pub struct JsConstantNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub value: JsDataInstance,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = ConstantNodeStub)]
impl JsConstantNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(value: JsDataInstance) -> JsConstantNodeStub {
        let cloned = value.value.clone();
        JsConstantNodeStub {
            value,
            stub: canvas::NodeStub::Constant(cloned).into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = OkOrCrashNodeStub)]
pub struct JsOkOrCrashNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub message: JsString,
}

#[wasm_bindgen(js_class = OkOrCrash)]
impl JsOkOrCrashNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(msg: JsString) -> JsOkOrCrashNodeStub {
        let rust_msg = String::from(&msg).into();
        JsOkOrCrashNodeStub {
            message: msg,
            stub: canvas::NodeStub::OkOrCrash { msg: rust_msg }.into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = CrashNodeStub)]
pub struct JsCrashNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub message: JsString,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = CrashNode)]
impl JsCrashNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(msg: JsString) -> JsCrashNodeStub {
        let rust_msg = String::from(&msg).into();
        JsCrashNodeStub {
            message: msg,
            stub: canvas::NodeStub::Crash { msg: rust_msg }.into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = UnreachableNodeStub)]
pub struct JsUnreachableNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub message: JsString,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = UnreachableNodeStub)]
impl JsUnreachableNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(msg: JsString) -> JsUnreachableNodeStub {
        let rust_msg = String::from(&msg).into();
        JsUnreachableNodeStub {
            message: msg,
            stub: canvas::NodeStub::Unreachable { msg: rust_msg }.into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = TodoNodeStub)]
pub struct JsTodoNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub message: JsString,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = TodoNodeStub)]
impl JsTodoNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(msg: JsString, inputs: usize) -> JsTodoNodeStub {
        let rust_msg = String::from(&msg).into();
        JsTodoNodeStub {
            message: msg,
            stub: canvas::NodeStub::Todo {
                msg: rust_msg,
                inputs: inputs as canvas::PinOrder,
            }
            .into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = CommentNodeStub)]
pub struct JsCommentNodeStub {
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub message: JsString,

    #[wasm_bindgen(readonly, getter_with_clone)]
    pub stub: JsNodeStub,
}

#[wasm_bindgen(js_class = CommentNodeStub)]
impl JsCommentNodeStub {
    #[wasm_bindgen(constructor)]
    pub fn new(msg: JsString, inputs: usize) -> JsCommentNodeStub {
        let rust_msg = String::from(&msg).into();
        JsCommentNodeStub {
            message: msg,
            stub: canvas::NodeStub::Comment {
                msg: rust_msg,
                inputs: inputs as canvas::PinOrder,
            }
            .into(),
        }
    }
}

#[derive(Debug)]
#[wasm_bindgen(js_name = NodeStubKind)]
pub enum JsNodeStubKind {
    File,
    SplitBy,
    Input,
    Drop,
    Output,
    Lowercase,
    Uppercase,
    Strip,
    Compare,
    Ordering,
    Regex,
    FindRecord,
    Map,
    List,
    IfElse,
    OkOrErr,
    Validate,
    SelectFirst,
    ExpectOne,
    ExpectSome,
    Match,
    Func,
    ParseDateTime,
    ParseMonetary,
    ParseInt,
    Constant,
    OkOrCrash,
    Crash,
    Unreachable,
    Todo,
    Comment,
}

/// Node in the canvas. This is actual instance of the placed node in a project.
#[derive(Debug)]
#[wasm_bindgen(js_name = Node)]
pub struct JsNode {
    project_uuid: Uuid,
    node_id: canvas::Id,
}

#[wasm_bindgen(js_class = Node)]
impl JsNode {
    /// Get the stub that will allow to create a new node with the same configuration.
    ///
    /// # Errors
    /// If the project or the node is no longer valid, this will return an error.
    #[wasm_bindgen(getter, js_name = stub)]
    pub fn stub(&self) -> Result<JsNodeStub, JsError> {
        let ws = wsr!();
        let maybe_project = ws.project_by_id(self.project_uuid);
        if let Some(project) = maybe_project {
            if let Some(node) = project.canvas().node(self.node_id) {
                Ok(node.stub.clone().into())
            } else {
                Err(InvalidHandleError.into())
            }
        } else {
            Err(InvalidHandleError.into())
        }
    }

    /// Get the output pin at the given position.
    #[wasm_bindgen(js_name = outAt)]
    pub fn out_at(&self, position: u32) -> Result<JsNodePin, JsError> {
        let ws = wsr!();
        let project = ws
            .project_by_id(self.project_uuid)
            .ok_or(InvalidHandleError)?;
        let node = project
            .canvas()
            .node(self.node_id)
            .ok_or(InvalidHandleError)?;
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

        let mut ws = wsw!();
        let project = ws.project_by_id_mut(self.project_uuid);
        if let Some(project) = project {
            project.remove_node(self.node_id)?;
            Ok(stub)
        } else {
            error!("Project not found. Was removed from work session. Project handle is invalid.");
            Err(InvalidHandleError.into())
        }
    }
}

/// Node iterator to fetch nodes from the canvas. This exists so to avoid
/// duplicating big arrays of nodes in memory.
#[derive(Debug)]
#[wasm_bindgen(js_name = NodeIter)]
pub struct JsNodeIter {
    detect_change: DetectChangedStack,

    project_uuid: Uuid,
    pos: usize,
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

        let ws = wsr!();
        let project = ws.project_by_id(self.project_uuid)?;
        let node = project.canvas().node_by_idx(self.pos as _)?;
        self.pos += 1;
        Some(JsNode {
            project_uuid: self.project_uuid,
            node_id: node.id,
        })
    }

    #[wasm_bindgen(getter, js_name = isValid)]
    pub fn is_valid(&self) -> bool {
        let ws = wsr!();
        if let Some(p) = ws.project_by_id(self.project_uuid) {
            p.detect_changed_stack() == self.detect_change
        } else {
            false
        }
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

#[wasm_bindgen(js_class = NodePin)]
impl JsNodePin {
    /// Get the data type of the pin.
    /// If the pin has unknown data type, this will return `undefined`.
    #[wasm_bindgen(getter, js_name = dataType)]
    pub fn data_type(&self) -> Option<JsDataType> {
        let ws = wsr!();
        let project = ws.project_by_id(self.project_uuid)?;
        let ty = project.pin_data_type(canvas::Pin {
            node_id: self.node_id,
            order: self.ordinal,
        })?;
        Some(ty.to_owned().into())
    }

    /// Whether the pin is an output pin (`true`). If it is an input pin, this returns `false`.
    #[wasm_bindgen(getter, js_name = isOutput)]
    pub fn is_output(&self) -> Option<bool> {
        let ws = wsr!();
        let project = ws.project_by_id(self.project_uuid)?;
        let node = project.canvas().node(self.node_id)?;
        if node.stub.is_valid_input_ordinal(self.ordinal) {
            Some(false)
        } else if node.stub.is_valid_output_ordinal(self.ordinal) {
            Some(true)
        } else {
            error!("Pin ordinal is invalid. Cannot determine if it is input or output.");
            None
        }
    }

    /// Get the position of the pin on the node.
    ///
    /// This is a converted ordinal, not absolute position in the node as per internal logic.
    /// This uniquely identifies input and output pins separately.
    #[wasm_bindgen(getter)]
    pub fn ordinal(&self) -> Result<canvas::PinOrder, JsError> {
        let is_output = self.is_output().ok_or(InvalidHandleError)?;

        let ws = wsr!();
        let project = ws
            .project_by_id(self.project_uuid)
            .ok_or(InvalidHandleError)?;
        let node = project
            .canvas()
            .node(self.node_id)
            .ok_or(InvalidHandleError)?;

        if is_output {
            node.stub.real_output_pin_idx(self.ordinal)
        } else {
            node.stub.real_input_pin_idx(self.ordinal)
        }
        .ok_or(PinNotFoundError.into())
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
        use canvas::PrimitiveType::*;
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

    /// Get the value as a JS value.
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
    #[wasm_bindgen(readonly, getter_with_clone)]
    pub value: Option<JsDataInstance>,
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
