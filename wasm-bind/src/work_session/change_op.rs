use super::*;

/// Change operation stub, that can be used to execute actual change operation,
/// which than will record all necessary information to the history stack.
#[derive(Debug, Clone)]
pub enum ChangeOpStub {
    AddNode {
        stub: Box<canvas::NodeStub>,
    },
    RemoveNode {
        id: canvas::Id,
    },
    AddEdge {
        edge: canvas::Edge,
    },
    RemoveEdge {
        edge: canvas::Edge,
    },
    AlterNodeMetadata {
        node_id: canvas::Id,
        key: JsString,
        new: JsValue,
    },
    AlterProjectMetadata {
        key: JsString,
        new: JsValue,
    },
}

/// Single change operation that was performed on the project.
/// This record allows to revert the existing change.
/// This is the smallest unit of action recordable in [ChangeItem].
#[derive(Debug, Clone)]
pub enum ChangeOp {
    /// Add a node to the canvas.
    AddNode {
        stub: Box<canvas::NodeStub>,
        id: canvas::Id,
    },

    /// Remove a node from the canvas.
    RemoveNode {
        removed_edges: Vec<canvas::Edge>,
        stub: Box<canvas::NodeStub>,
        meta: Metadata,
        id: canvas::Id,
    },

    /// Add an edge to the canvas.
    AddEdge { edge: canvas::Edge },

    /// Remove an edge from the canvas.
    RemoveEdge { edge: canvas::Edge },

    /// Alter metadata of a node.
    AlterNodeMetadata {
        /// Node in which the metadata was changed.
        node_id: canvas::Id,

        /// A key to the metadata that was changed.
        key: JsString,

        /// New value assigned.
        new: JsValue,

        /// Backup of the previous metadata value.
        backup: JsValue,
    },

    /// Alter metadata of the project.
    AlterProjectMetadata {
        /// A key to the metadata that was changed.
        key: JsString,

        /// New value assigned.
        new: JsValue,

        /// Backup of the previous metadata value.
        backup: JsValue,
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
    pub(super) fn apply(
        project: &mut WorkSessionProject,
        change_op_stub: ChangeOpStub,
    ) -> ChangeOp {
        use ChangeOpStub::*;

        const EXPECT_FOUND_NODE: &str = "node not found, though operation was recorded";
        const EXPECT_FOUND_EDGE: &str = "edge not found, though operation was recorded";

        match change_op_stub {
            AddNode { stub } => Self::add_node(project, (*stub).clone()).into(),
            RemoveNode { id } => Self::remove_node(project, id)
                .expect(EXPECT_FOUND_NODE)
                .into(),
            AddEdge { edge } => Self::add_edge(project, edge)
                .expect(EXPECT_FOUND_EDGE)
                .into(),
            RemoveEdge { edge } => Self::remove_edge(project, edge)
                .expect(EXPECT_FOUND_EDGE)
                .into(),
            AlterNodeMetadata { node_id, key, new } => {
                Self::alter_node_metadata(project, node_id, key, new)
                    .expect(EXPECT_FOUND_NODE)
                    .into()
            }
            AlterProjectMetadata { key, new } => {
                Self::alter_project_metadata(project, key, new).into()
            }
        }
    }

    /// Revert the change operation from the project.
    /// This is the opposite of [ChangeOp::apply].
    pub(super) fn revert(self, project: &mut WorkSessionProject) {
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
                key,
                new: _,
                backup,
            } => {
                let meta = &mut project
                    .project
                    .canvas_mut()
                    .node_mut(node_id)
                    .expect(EXPECT_FOUND_NODE)
                    .meta;
                meta.insert(key.into(), backup);
            }
            AlterProjectMetadata {
                key,
                new: _,
                backup,
            } => {
                let meta = &mut project.project.meta;
                meta.insert(key.into(), backup);
            }
        }
    }

    pub(super) fn add_node(project: &mut WorkSessionProject, stub: canvas::NodeStub) -> AddNode {
        let id = project
            .project
            .canvas_mut()
            .add_node(stub.clone(), Default::default());
        AddNode { stub, id }
    }

    pub(super) fn remove_node(
        project: &mut WorkSessionProject,
        id: canvas::Id,
    ) -> Result<RemoveNode, NodeNotFoundError> {
        let (node, edges) = project.project.canvas_mut().remove_node(id)?;
        Ok(RemoveNode { node, edges })
    }

    pub(super) fn add_edge(
        project: &mut WorkSessionProject,
        edge: canvas::Edge,
    ) -> Result<AddEdge, NodeNotFoundError> {
        project.project.canvas_mut().add_edge(edge)?;
        Ok(AddEdge { edge })
    }

    pub(super) fn remove_edge(
        project: &mut WorkSessionProject,
        edge: canvas::Edge,
    ) -> Result<RemoveEdge, EdgeNotFoundError> {
        project.project.canvas_mut().remove_edge(edge)?;
        Ok(RemoveEdge { edge })
    }

    pub(super) fn alter_node_metadata(
        project: &mut WorkSessionProject,
        node_id: canvas::Id,
        key: JsString,
        new: JsValue,
    ) -> Result<AlterNodeMetadata, NodeNotFoundError> {
        let meta = &mut project
            .project
            .canvas_mut()
            .node_mut(node_id)
            .ok_or(NodeNotFoundError(node_id))?
            .meta;

        let backup = meta
            .insert(key.clone().into(), new.clone())
            .unwrap_or(JsValue::UNDEFINED);

        Ok(AlterNodeMetadata {
            node_id,
            key,
            new,
            backup,
        })
    }

    pub(super) fn alter_project_metadata(
        project: &mut WorkSessionProject,
        key: JsString,
        new: JsValue,
    ) -> AlterProjectMetadata {
        let meta = &mut project.project.meta;
        let backup = meta
            .insert(key.clone().into(), new.clone())
            .unwrap_or(JsValue::UNDEFINED);

        AlterProjectMetadata { key, new, backup }
    }
}

#[derive(Debug)]
pub(super) struct AddNode {
    pub stub: canvas::NodeStub,
    pub id: canvas::Id,
}

impl From<AddNode> for ChangeOp {
    fn from(op: AddNode) -> Self {
        ChangeOp::AddNode {
            stub: Box::new(op.stub),
            id: op.id,
        }
    }
}

#[derive(Debug)]
pub(super) struct RemoveNode {
    pub node: canvas::Node<Metadata>,
    pub edges: Vec<canvas::Edge>,
}

impl From<RemoveNode> for ChangeOp {
    fn from(op: RemoveNode) -> Self {
        ChangeOp::RemoveNode {
            removed_edges: op.edges,
            stub: Box::new(op.node.stub),
            meta: op.node.meta,
            id: op.node.id,
        }
    }
}
#[derive(Debug)]
pub(super) struct AlterNodeMetadata {
    pub node_id: canvas::Id,
    pub key: JsString,
    pub new: JsValue,
    pub backup: JsValue,
}

impl From<AlterNodeMetadata> for ChangeOp {
    fn from(op: AlterNodeMetadata) -> Self {
        ChangeOp::AlterNodeMetadata {
            node_id: op.node_id,
            key: op.key,
            new: op.new,
            backup: op.backup,
        }
    }
}

#[derive(Debug)]
pub(super) struct AddEdge {
    pub edge: canvas::Edge,
}

impl From<AddEdge> for ChangeOp {
    fn from(op: AddEdge) -> Self {
        ChangeOp::AddEdge { edge: op.edge }
    }
}

#[derive(Debug)]
pub(super) struct RemoveEdge {
    pub edge: canvas::Edge,
}

impl From<RemoveEdge> for ChangeOp {
    fn from(op: RemoveEdge) -> Self {
        ChangeOp::RemoveEdge { edge: op.edge }
    }
}

#[derive(Debug)]
pub(super) struct AlterProjectMetadata {
    pub key: JsString,
    pub new: JsValue,
    pub backup: JsValue,
}

impl From<AlterProjectMetadata> for ChangeOp {
    fn from(op: AlterProjectMetadata) -> Self {
        ChangeOp::AlterProjectMetadata {
            key: op.key,
            new: op.new,
            backup: op.backup,
        }
    }
}
