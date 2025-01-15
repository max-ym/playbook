use smallvec::SmallVec;

use crate::canvas::{Edge, Id, Node, PrimitiveType, PrimitiveTypeConst, Value};

type NodeId = Id;
type EdgeId = Id;

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
    inputs: SmallVec<[EdgeId; 1]>,
    outputs: SmallVec<[EdgeId; 1]>,

    /// Sorted array of all data types for the edges in the plan.
    data_types: Vec<EdgeDataType>,
}

#[derive(Debug)]
struct EdgeDataType {
    edge_id: EdgeId,
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
pub struct Context<'canvas, NodeMeta> {
    nodes: &'canvas [Node<NodeMeta>],
    edges: &'canvas [Edge],

    /// Sorted array of all predicates resolved into predicate plans.
    /// Sorting allowes for binary search to find the predicate plan.
    predicate_plans: &'canvas [PredicatePlan<'canvas>],
}

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
}

impl<'canvas, NodeMeta> ValidationPlan<'canvas, NodeMeta> {
    /// Create a new validation plan(s) to include all steps leading to this node, including
    /// the node itself. If the node can be traced back to several flows, each flow will
    /// be included in the separate plan.
    pub fn node_backtrace(
        ctx: Context<'canvas, NodeMeta>,
        idx: NodeId,
    ) -> BacktraceResult<'canvas, NodeMeta> {
        todo!()
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
    pub fn edge_data_type(&self, edge: EdgeId) -> Option<&PrimitiveType> {
        self.data_types
            .binary_search_by(|x| x.edge_id.cmp(&edge))
            .ok()
            .map(|idx| &self.data_types[idx].data_type)
    }
}

pub struct BacktraceResult<'canvas, NodeMeta> {
    valid_plans: Vec<ValidationPlan<'canvas, NodeMeta>>,
    errors: Vec<NodeBacktraceError>,
}

pub struct TypeDeterminatorError {
    invalid_types: Vec<TypeConflict>,
}

pub enum NodeBacktraceError {
    /// The node is not a part of the validation plan.
    NotFound,

    /// The node is not connected to the data source (input, or root node).
    NoInput,

    /// Node is a part of a cycle.
    Cycle(Vec<NodeId>),
}

/// Type conflict error that is created during the type determination stage.
#[derive(Debug)]
pub struct TypeConflict {
    /// Edge that has a type conflict.
    edge: EdgeId,

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
