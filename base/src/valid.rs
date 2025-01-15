use smallvec::SmallVec;

use crate::canvas::{Edge, Id, Node, Value};

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
    inputs: SmallVec<[EdgeId; 1]>,
    outputs: SmallVec<[EdgeId; 1]>,
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

impl<NodeMeta> ValidationPlan<'_, NodeMeta> {
    /// Create a new validation plan to include all steps leading to this node, including
    /// the node itself.
    pub fn node_backtrace(ctx: Context<NodeMeta>, idx: NodeId) -> Result<Self, Todo> {
        todo!()
    }

    /// Put a value set into the validation plan and return the resulting values.
    /// The number of input values should match the number of input pins.
    ///
    /// The function returns the output values into the `out` slice. It thus can be
    /// reused as a buffer to avoid unnecessary allocations for subsequent calls with
    /// different input values. The function will return error if the `out` buffer is of wrong
    /// size.
    pub fn validate_value_set(
        &self,
        values: impl IntoIterator<Item = Value>,
        out: &mut [Value],
    ) -> Result<(), Todo> {
        todo!()
    }
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
