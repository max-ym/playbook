use log::trace;
use smallvec::SmallVec;

use crate::canvas::{Edge, Id, Node, Pin, PrimitiveType, PrimitiveTypeConst, Value};

type NodeId = Id;

type EdgeIdx = usize;

/// Chains are individual groups of flows that connect one input,
/// pass the data through a series of transformations (nodes), and at the end
/// output the result. The chain thus allows to separate the data processing
/// into smaller, more manageable parts.
///
/// The chain is terminated as soon as the next output is split between multiple
/// nodes. At this point the data outputted by the chain is collected and passed
/// to individual next nodes, which in turn form their own chains.
///
/// Chains in the context of validation plan is a single manageable step of validation.
#[derive(Debug)]
struct Chain {
    nodes: Vec<NodeId>,
}

type ChainId = usize;

/// Validation plan is a collection of chains that are executed in a specific order.
#[derive(Debug)]
pub struct ValidationPlan<'canvas, NodeMeta> {
    nodes: Vec<&'canvas Node<NodeMeta>>,
    edges: Vec<&'canvas Edge>,

    chains: Vec<Chain>,
    steps: Vec<ChainId>,

    /// The edges that are the data inputs to the validation plan.
    /// For the plans that contain root nodes (which are effectively generating new data),
    /// the inputs here are the output edges of those root nodes.
    ///
    /// Input data types are crucial for the validation plan to be able to validate the data,
    /// and should be present in `data_types` field from the creation of the plan, even before
    /// type deduction stage.
    inputs: SmallVec<[Pin; 1]>,
    outputs: SmallVec<[Pin; 1]>,

    /// Sorted array of all data types for the edges in the plan.
    data_types: Vec<EdgeDataType>,
}

#[derive(Debug)]
struct EdgeDataType {
    edge_idx: EdgeIdx,
    data_type: PrimitiveType,
}

/// A helper struct that is used to store information about how
/// a predicate should be executed.
#[derive(Debug)]
pub struct PredicatePlan<'canvas> {
    /// ID of the node that is the predicate.
    id: NodeId,

    /// Validation plan that represents the predicate.
    validation_plan: ValidationPlan<'canvas, ()>,
}

impl PartialEq for PredicatePlan<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for PredicatePlan<'_> {}

impl PartialOrd for PredicatePlan<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PredicatePlan<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

/// Context is a helper struct that is used to store all information needed to
/// build a validation plan.
#[derive(Debug)]
pub struct Context<'canvas, NodeMeta> {
    nodes: &'canvas [Node<NodeMeta>],
    edges: &'canvas [Edge],

    /// Sorted array of all predicates resolved into predicate plans.
    /// Sorting allowes for binary search to find the predicate plan.
    predicate_plans: &'canvas [PredicatePlan<'canvas>],
}

impl<T> std::clone::Clone for Context<'_, T> {
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes,
            edges: self.edges,
            predicate_plans: self.predicate_plans,
        }
    }
}

impl<T> std::marker::Copy for Context<'_, T> {}

impl<'canvas, NodeMeta> Context<'canvas, NodeMeta> {
    /// Create a new context with the given nodes, edges and predicate plans.
    pub fn new(
        nodes: &'canvas [Node<NodeMeta>],
        edges: &'canvas [Edge],
        predicate_plans: &'canvas [PredicatePlan],
    ) -> Self {
        if cfg!(debug_assertions) {
            assert!(predicate_plans.is_sorted());

            // All edges have matching nodes.
            for edge in edges {
                assert!((edge.from.node_id.get() as usize) < nodes.len());
                assert!((edge.to.node_id.get() as usize) < nodes.len());
            }

            // All plans have matching nodes.
            for plan in predicate_plans {
                assert!((plan.id.get() as usize) < nodes.len());
            }

            // All predicate plans are backed by "predicate" nodes.
            for plan in predicate_plans {
                assert!(nodes[plan.id.get() as usize].is_predicate());
            }
        }

        Self {
            nodes,
            edges,
            predicate_plans,
        }
    }

    fn nodes_connected_to(&self, node_id: NodeId, connected: &mut SmallVec<[EdgeAndNode; 32]>) {
        connected.clear();

        let place = self.binary_search_edge_place(node_id);
        while self.edges[place].from.node_id == node_id {
            let node = self.edges[place].from.node_id;
            let edge = EdgeIdx::from(place);
            connected.push(EdgeAndNode { edge, node });
        }
    }

    /// Use binary search to find the place of the edge for a given node in a
    /// sorted array of edges. This either actually finds the real edge, but generally
    /// it returns a place where the node's edges would be, if they were present.
    fn binary_search_edge_place(&self, node_id: NodeId) -> usize {
        match self.edges.binary_search(&Edge::binary_search_from(node_id)) {
            Ok(v) => v,
            Err(v) => v,
        }
    }

    /// Get predicate plan for the given node.
    fn predicate_plan_for(&self, node_id: NodeId) -> &PredicatePlan<'canvas> {
        let place = self
            .predicate_plans
            .binary_search_by(|plan| plan.id.cmp(&node_id))
            .unwrap();
        &self.predicate_plans[place]
    }

    fn node(&self, node_id: NodeId) -> &Node<NodeMeta> {
        let pos = self
            .nodes
            .binary_search_by(|node| node.id.cmp(&node_id))
            .unwrap();
        &self.nodes[pos]
    }

    fn edge_idx(&self, edge: Edge) -> Option<EdgeIdx> {
        self.edges.binary_search_by(|e| e.cmp(&edge)).ok()
    }
}

#[derive(Debug, Clone, Copy)]
struct EdgeAndNode {
    edge: EdgeIdx,
    node: NodeId,
}

impl<'canvas, NodeMeta> ValidationPlan<'canvas, NodeMeta> {
    /// Create a new validation path(s) to include all steps leading to this node, including
    /// the node itself. If the node can be traced back to several flows, each flow will
    /// be included in the separate path.
    pub fn node_backtrace(
        ctx: Context<'canvas, NodeMeta>,
        idx: NodeId,
    ) -> Result<BacktraceResult<'canvas, NodeMeta>, NonexistentElementError> {
        let original_node_idx = idx;
        let mut errors = SmallVec::<[_; 32]>::new();

        if ctx.nodes.get(original_node_idx.get() as usize).is_none() {
            return Err(NonexistentElementError);
        };

        type Path = SmallVec<[EdgeAndNode; 32]>;

        // Array of paths. Each path is a separate flow that leads to the node.
        let mut paths = SmallVec::<[_; 32]>::new();
        let mut connected_nodes_buf = SmallVec::<[_; 32]>::new();
        let mut unfinished_path_idx = SmallVec::<[usize; 32]>::new();

        // Find the edges that lead to the original node and make them the starting points
        // for the paths.
        ctx.nodes_connected_to(original_node_idx, &mut connected_nodes_buf);
        unfinished_path_idx.reserve(connected_nodes_buf.len());
        for &edge_node in &connected_nodes_buf {
            let mut path = SmallVec::<[_; 32]>::new();
            path.push(edge_node);
            unfinished_path_idx.push(paths.len());
            paths.push(path);
        }

        // Traverse the unfinished flows from the original node.
        while !unfinished_path_idx.is_empty() {
            // Unfinished paths are the paths that have not reached the end yet.
            // We look into them to see if they have more nodes to connect.
            for path_idx in unfinished_path_idx.clone() {
                // Closure to add a node to the path, validating for cycles.
                let mut add_node = |path_idx, path: &mut Path, edge_node: EdgeAndNode| {
                    if edge_node.node == original_node_idx {
                        // The loop is detected. This is a cycle for this path.
                        errors.push(NodeBacktraceError::Cycle(std::mem::take(path).into_vec()));
                        // `into_vec` clear the path. Clearing marks the path as invalid.
                    } else {
                        path.push(edge_node);
                        unfinished_path_idx.push(path_idx);
                    }
                };

                let last = *paths[path_idx]
                    .last()
                    .expect("all paths have at least one node");
                ctx.nodes_connected_to(last.node, &mut connected_nodes_buf);

                let first_node = if let Some(first_node) = connected_nodes_buf.pop() {
                    first_node
                } else {
                    // The node is not connected to anything. This is the end of the path.
                    continue;
                };

                // Rest of the found nodes have different logic than the first one.
                // We need to duplicate the path, as it will diverge at this point for
                // each found node.
                for &remaining_node in &connected_nodes_buf {
                    let mut new_path = paths[path_idx].clone();
                    add_node(paths.len(), &mut new_path, remaining_node);
                    paths.push(new_path);
                }

                // Add the first node to the original path.
                add_node(path_idx, &mut paths[path_idx], first_node);
            }
        }

        let mut result = Vec::with_capacity(paths.len());
        // Check that all paths lead to a root node.
        for path in paths {
            if let Some(&last) = path.last() {
                let is_root = ctx.nodes.get(last.node.get() as usize).unwrap().is_root();
                if !is_root {
                    errors.push(NodeBacktraceError::NoInput(path.into_vec()));
                } else {
                    // The path is valid.
                    result.push(Self::new_for_edge_node_rev(ctx, &path));
                }
            } else {
                // Path was marked as invalid, we can ignore it.
            }
        }

        Ok(BacktraceResult {
            valid_paths: result,
            errors: errors.into_vec(),
        })
    }

    /// Create a plan from all nodes and edges from the context that are part of the given path.
    /// This accepts reversed order, generated from backtracing from selected node up to
    /// the root.
    fn new_for_edge_node_rev(ctx: Context<'canvas, NodeMeta>, edge_node: &[EdgeAndNode]) -> Self {
        let mut nodes = Vec::with_capacity(edge_node.len());
        let mut edges = Vec::with_capacity(edge_node.len());
        for edge_node in edge_node.iter().rev() {
            nodes.push(&ctx.nodes[edge_node.node.get() as usize]);
            edges.push(&ctx.edges[edge_node.edge]);
        }

        trace!("Make a single chain to represent the path");
        let chain = Chain {
            nodes: nodes.iter().map(|node| node.id).collect(),
        };

        trace!("Define inputs for the plan");
        let root = nodes.first().unwrap();
        debug_assert!(root.is_root());
        let after_root = nodes.iter().skip(1).next()
            .expect("we cannot compute without nodes. This is a bug, as the node array should be verified before this.");
        let inputs = Self::count_node_io(ctx, after_root.id).0;
        let last = nodes
            .last()
            .expect("there are elements per operations above");
        let outputs = Self::count_node_io(ctx, last.id).1;

        trace!("Collect inputs and outputs as pins");
        let mut inputs_vec = SmallVec::with_capacity(inputs);
        for i in 0..inputs {
            inputs_vec.push(Pin {
                node_id: after_root.id,
                order: i as _,
            });
        }
        let mut outputs_vec = SmallVec::with_capacity(outputs);
        for i in 0..outputs {
            outputs_vec.push(Pin {
                node_id: last.id,
                order: i as _,
            });
        }

        trace!("Set input data types");
        // Since our plan starts with a root node, we set "Record" data type for all inputs.
        let mut data_types = Vec::with_capacity(inputs);
        for i in 0..inputs {
            data_types.push(EdgeDataType {
                edge_idx: ctx
                    .edge_idx(Edge {
                        from: Pin::only_node_id(root.id), // this will work as root node has only one output
                        to: inputs_vec[i],
                    })
                    .unwrap(),
                data_type: PrimitiveType::Record,
            });
        }

        Self {
            nodes,
            edges,
            chains: vec![chain],
            steps: vec![],
            inputs: inputs_vec,
            outputs: outputs_vec,
            data_types,
        }
    }

    fn count_node_io(ctx: Context<NodeMeta>, node_id: NodeId) -> (usize, usize) {
        let node = ctx.node(node_id);
        if node.is_predicate() {
            let plan = &ctx.predicate_plan_for(node_id).validation_plan;
            (plan.inputs.len(), plan.outputs.len())
        } else if let (Some(inputs), Some(outputs)) = (node.static_inputs(), node.static_outputs())
        {
            (inputs, outputs)
        } else {
            unreachable!("Reaching this means there are no mechanics defined to count IO for the node. Missing impl?");
        }
    }

    /// Create a new validation plan describing all flows in a canvas, to validate it as
    /// a whole.
    pub fn canvas_plan(ctx: Context<'canvas, NodeMeta>) -> Result<Self, Todo> {
        todo!()
    }

    /// Put a value set into the validation plan and return the resulting values.
    /// The number of input values should match the number of input pins.
    ///
    /// The function returns the output values into the `out` slice. It thus can be
    /// reused as a buffer to avoid unnecessary allocations for subsequent calls with
    /// different input values. The function will return error if the `out` buffer is of wrong
    /// size.
    ///
    /// If the data type determination stage was not run, the function will run it.
    pub fn validate_value_set(
        &self,
        values: impl IntoIterator<Item = Value>,
        out: &mut [Value],
    ) -> Result<(), Todo> {
        todo!()
    }

    /// Run the the step determination stage. This plans the correct order of the chains execution.
    pub fn determine_steps(&mut self) -> Result<(), Todo> {
        todo!()
    }

    /// Run type determination stage on the validation plan.
    /// This fills in all the data types for the edges in the plan, allowing for
    /// querying the types of the data that flows through the plan.
    ///
    /// This requires [determine_steps](Self::determine_steps) to be run first.
    /// If it was not, this function will run it.
    pub fn determine_types(&mut self) -> Result<(), TypeDeterminatorError> {
        todo!()
    }

    /// Get the data type of the edge if it was determined.
    /// Returns `None` if the type was not determined.
    ///
    /// You should run [determine_types](Self::determine_types) to fill in the types.
    pub fn edge_data_type(&self, edge: EdgeIdx) -> Option<&PrimitiveType> {
        self.data_types
            .binary_search_by(|x| x.edge_idx.cmp(&edge))
            .ok()
            .map(|idx| &self.data_types[idx].data_type)
    }
}

pub struct BacktraceResult<'canvas, NodeMeta> {
    valid_paths: Vec<ValidationPlan<'canvas, NodeMeta>>,
    errors: Vec<NodeBacktraceError>,
}

pub struct TypeDeterminatorError {
    invalid_types: Vec<TypeConflict>,
}

pub enum NodeBacktraceError {
    /// The node is not connected to the data source (input, or root node).
    NoInput(Vec<EdgeAndNode>),

    /// Node is a part of a cycle.
    Cycle(Vec<EdgeAndNode>),
}

#[derive(Debug)]
pub struct NonexistentElementError;

/// Type conflict error that is created during the type determination stage.
#[derive(Debug)]
pub struct TypeConflict {
    /// Edge that has a type conflict.
    edge: EdgeIdx,

    /// Found type for the edge per output node.
    found: PrimitiveType,

    /// Expected types for the edge. These are defined statically as constants.
    /// If the expected types are not set statically, this field is `None`.
    static_expected: Option<&'static [PrimitiveTypeConst]>,

    /// Expected types for the edge. These are defined dynamically by the nodes.
    /// These are supplemented by the static expected types.
    /// This can be empty if static expected types are determined.
    expected: Vec<PrimitiveType>,
}

/// Remove later. A placeholder for undefined stuff.
pub struct Todo;

/// Information about all collected chains and their dependencies.
pub struct Chains {
    chains: Vec<Chain>,
    chain_dependencies: Vec<ChainDependOn>,
}

#[derive(Debug, Clone, Copy)]
struct ChainDependOn {
    chain: ChainId,
    depends_on: ChainId,
}
