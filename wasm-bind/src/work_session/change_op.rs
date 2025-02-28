use super::*;

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
    /// Apply this change operation to the project, updating the operation
    /// with altered new values (like new ID for a node).
    ///
    /// # Panic
    /// This method panics if the operation cannot be applied, which cannot
    /// happen if the operation was recorded correctly and when all changes are
    /// tracked correctly.
    pub(super) fn apply(&mut self, project: &mut Project) {
        use ChangeOp::*;

        const EXPECT_FOUND_NODE: &str = "node not found, though operation was recorded";
        const EXPECT_FOUND_EDGE: &str = "edge not found, though operation was recorded";

        let new = match self {
            AddNode { stub, .. } => Self::add_node(project, (**stub).clone()).into(),
            RemoveNode { id, .. } => Self::remove_node(project, *id)
                .expect(EXPECT_FOUND_NODE)
                .into(),
            AddEdge { edge } => Self::add_edge(project, *edge)
                .expect(EXPECT_FOUND_EDGE)
                .into(),
            RemoveEdge { edge } => Self::remove_edge(project, *edge)
                .expect(EXPECT_FOUND_EDGE)
                .into(),
            AlterNodeMetadata {
                node_id, key, new, ..
            } => Self::alter_node_metadata(project, *node_id, key.clone(), new.clone())
                .expect(EXPECT_FOUND_NODE)
                .into(),
            AlterProjectMetadata { key, new, .. } => {
                Self::alter_project_metadata(project, key.clone(), new.clone()).into()
            }
        };
        *self = new;
    }

    /// Revert the change operation from the project.
    /// This is the opposite of [ChangeOp::apply].
    pub(super) fn revert(&self, project: &mut Project) {
        use ChangeOp::*;

        const EXPECT_FOUND_NODE: &str = "node not found, though operation was recorded";
        const EXPECT_FOUND_EDGE: &str = "edge not found, though operation was recorded";

        match self {
            AddNode { id, .. } => {
                project
                    .canvas_mut()
                    .remove_node(*id)
                    .expect(EXPECT_FOUND_NODE);
            }
            RemoveNode {
                removed_edges,
                stub,
                meta,
                id: _,
            } => {
                project
                    .canvas_mut()
                    .add_node((**stub).clone(), meta.clone());

                for edge in removed_edges {
                    project
                        .canvas_mut()
                        .add_edge(*edge)
                        .expect(EXPECT_FOUND_EDGE);
                }
            }
            AddEdge { edge } => {
                project
                    .canvas_mut()
                    .remove_edge(*edge)
                    .expect(EXPECT_FOUND_EDGE);
            }
            RemoveEdge { edge } => {
                project
                    .canvas_mut()
                    .add_edge(*edge)
                    .expect(EXPECT_FOUND_EDGE);
            }
            AlterNodeMetadata {
                node_id,
                key,
                new: _,
                backup,
            } => {
                let meta = &mut project
                    .canvas_mut()
                    .node_mut(*node_id)
                    .expect(EXPECT_FOUND_NODE)
                    .meta;
                meta.insert(key.clone().into(), backup.clone());
            }
            AlterProjectMetadata {
                key,
                new: _,
                backup,
            } => {
                let meta = &mut project.meta;
                meta.insert(key.clone().into(), backup.clone());
            }
        }
    }

    pub(super) fn add_node(project: &mut Project, stub: canvas::NodeStub) -> AddNode {
        let id = project
            .canvas_mut()
            .add_node(stub.clone(), Default::default());
        AddNode { stub, id }
    }

    pub(super) fn remove_node(
        project: &mut Project,
        id: canvas::Id,
    ) -> Result<RemoveNode, NodeNotFoundError> {
        let (node, edges) = project.canvas_mut().remove_node(id)?;
        Ok(RemoveNode { node, edges })
    }

    pub(super) fn add_edge(
        project: &mut Project,
        edge: canvas::Edge,
    ) -> Result<AddEdge, NodeNotFoundError> {
        project.canvas_mut().add_edge(edge)?;
        Ok(AddEdge { edge })
    }

    pub(super) fn remove_edge(
        project: &mut Project,
        edge: canvas::Edge,
    ) -> Result<RemoveEdge, EdgeNotFoundError> {
        project.canvas_mut().remove_edge(edge)?;
        Ok(RemoveEdge { edge })
    }

    pub(super) fn alter_node_metadata(
        project: &mut Project,
        node_id: canvas::Id,
        key: JsString,
        new: JsValue,
    ) -> Result<AlterNodeMetadata, NodeNotFoundError> {
        let meta = &mut project
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
        project: &mut Project,
        key: JsString,
        new: JsValue,
    ) -> AlterProjectMetadata {
        let meta = &mut project.meta;
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
