use std::{
    fmt::{Display, Formatter},
    ops::{Deref, DerefMut, Range},
};

use bigdecimal::BigDecimal;
use compact_str::CompactString;
use lazy_regex::Regex;
use log::{trace, warn};
use rand::rngs::SmallRng;
use smallvec::SmallVec;
use thiserror::Error;

/// Type validation and resolution of the data in canvas.
pub mod valid;

#[derive(Debug)]
pub struct Canvas<NodeMeta> {
    pub(crate) nodes: Vec<Node<NodeMeta>>,
    pub(crate) edges: Vec<Edge>,
    pub(crate) predicates: Vec<CanvasPredicate<NodeMeta>>,

    rnd: SmallRng,

    /// IDs of the root nodes of the canvas.
    /// In a valid project, from these nodes the execution of the flow starts.
    /// Some nodes cannot function as a starting point of execution, but they are still
    /// considered root nodes, as they are not connected to any other node, and then
    /// they look as a starting point of execution in their subflow.
    ///
    /// Note that this array stores all nodes that do not have a parent, even if they
    /// cannot function as a starting point of execution in the valid project.
    /// This array is later used by the validation to ensure that all nodes are reachable.
    pub(crate) root_nodes: Vec<Id>,

    pub(crate) valid: valid::Validator,
}

/// The type used for [Id] representation. This transitively defines the maximum number of
/// nodes and edges and thus the index type to use on their arrays.
type IdInnerType = u32;

pub(crate) type NodeIdx = IdInnerType;
pub(crate) type EdgeIdx = IdInnerType;

impl<NodeMeta> Canvas<NodeMeta> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            predicates: Vec::new(),
            root_nodes: Vec::new(),
            valid: valid::Validator::new(),
            rnd: {
                use rand::SeedableRng;
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .map(SmallRng::seed_from_u64)
                    .unwrap_or_else(|_| SmallRng::seed_from_u64(0))
            },
        }
    }

    pub fn add_node(&mut self, stub: NodeStub, meta: NodeMeta) -> Id {
        let last = self.nodes.last().map(|n| n.id).unwrap_or(Id(0));
        let id = Id::new_node_after(last, &mut self.rnd)
            .expect("node ID generation failed, maybe ID pool is used up");

        trace!("add node {id} with {} pins to the canvas. {stub:?}", stub.total_pin_count());
        let pin_cnt = stub.total_pin_count();
        self.root_nodes.push(id);
        self.nodes.push(Node { id, stub, meta });
        
        self.valid.add_node(id, pin_cnt as usize);
        id
    }

    pub fn node(&self, id: Id) -> Option<&Node<NodeMeta>> {
        self.node_id_to_idx(id).map(|idx| &self.nodes[idx as usize])
    }

    pub fn node_mut(&mut self, id: Id) -> Option<&mut Node<NodeMeta>> {
        self.node_id_to_idx(id)
            .map(move |idx| &mut self.nodes[idx as usize])
    }

    pub fn node_by_idx(&self, idx: NodeIdx) -> Option<&Node<NodeMeta>> {
        self.nodes.get(idx as usize)
    }

    pub fn root_nodes_inner(&self) -> impl Iterator<Item = NodeIdx> + '_ {
        self.root_nodes.iter().map(move |&id| {
            self.node_id_to_idx(id)
                .expect("root node should exist in the node list")
        })
    }

    fn node_edge_io_ranges(&self, node_id: Id) -> Option<(Range<usize>, Range<usize>)> {
        let range_out = {
            let start_edge_out = Edge {
                from: OutputPin(Pin { node_id, order: 0 }),
                to: InputPin(Pin {
                    node_id: Id(0),
                    order: 0,
                }),
            };
            let end_edge_out = Edge {
                from: OutputPin(Pin {
                    node_id,
                    order: PinOrder::MAX,
                }),
                to: InputPin(Pin {
                    node_id: Id(0),
                    order: 0,
                }),
            };

            let start = match self.edges.binary_search(&start_edge_out) {
                Ok(idx) => idx,
                Err(idx) => idx,
            };
            let end = match self.edges.binary_search(&end_edge_out) {
                Ok(idx) => idx,
                Err(idx) => idx,
            };
            let range = start..end;
            if range.is_empty() {
                trace!("no edges found for node {node_id} output");
            } else {
                trace!("edges found for node {node_id} output: {range:?}");
            }
            range
        };

        let range_in = {
            let start_edge_in = Edge {
                from: OutputPin(Pin {
                    node_id: Id(0),
                    order: 0,
                }),
                to: InputPin(Pin { node_id, order: 0 }),
            };
            let end_edge_in = Edge {
                from: OutputPin(Pin {
                    node_id: Id(0),
                    order: 0,
                }),
                to: InputPin(Pin {
                    node_id,
                    order: PinOrder::MAX,
                }),
            };

            let start = match self.edges.binary_search(&start_edge_in) {
                Ok(idx) => idx,
                Err(idx) => idx,
            };
            let end = match self.edges.binary_search(&end_edge_in) {
                Ok(idx) => idx,
                Err(idx) => idx,
            };
            let range = start..end;
            if range.is_empty() {
                trace!("no edges found for node {node_id} input");
            } else {
                trace!("edges found for node {node_id} input: {range:?}");
            }
            range
        };

        Some((range_in, range_out))
    }

    fn node_edge_io_slices(&self, node_id: Id) -> Option<(&[Edge], &[Edge])> {
        let (inp, out) = self.node_edge_io_ranges(node_id)?;
        Some((&self.edges[inp], &self.edges[out]))
    }

    fn node_edge_io_slices_mut(&mut self, node_id: Id) -> Option<(&mut [Edge], &mut [Edge])> {
        let (inp, out) = self.node_edge_io_ranges(node_id)?;
        let (i, o) = {
            let reversed = if inp.start < out.start { false } else { true };
            macro_rules! calc {
                ($low:expr, $high:expr) => {{
                    let (a, rest) = self.edges.split_at_mut($low.start);
                    let (_, b) = rest.split_at_mut($high.end + 1 - $low.start);

                    let a_len = $high.end - $high.start + 1;
                    let b_len = $low.end - $low.start + 1;
                    (&mut a[0..a_len], &mut b[0..b_len])
                }};
            }
            if reversed {
                // input > output
                let (a, b) = calc!(out, inp);
                (b, a)
            } else {
                // output > input
                calc!(inp, out)
            }
        };
        Some((i, o))
    }

    /// Iterator over all edges connected to the node.
    /// Returns [None] if the node does not exist.
    pub fn node_edge_io_iter(&self, node_id: Id) -> Option<impl Iterator<Item = Edge> + '_> {
        let (slice_in, slice_out) = self.node_edge_io_slices(node_id)?;
        Some(slice_in.iter().chain(slice_out.iter()).copied())
    }

    pub(crate) fn node_id_to_idx(&self, id: Id) -> Option<NodeIdx> {
        // Node array is sorted, so we can use binary search.
        debug_assert!(self.nodes.is_sorted_by_key(|n| n.id));
        self.nodes
            .binary_search_by_key(&id, |n| n.id)
            .ok()
            .map(|v| v as NodeIdx)
    }

    pub fn remove_node(
        &mut self,
        id: Id,
    ) -> Result<(Node<NodeMeta>, Vec<Edge>), NodeNotFoundError> {
        // Remove all edges that are connected to the node.
        let (inp, out) = self.node_edge_io_ranges(id).ok_or(NodeNotFoundError(id))?;
        let (high, low) = if inp.start > out.start {
            (inp, out)
        } else {
            (out, inp)
        };
        let mut edges = Vec::with_capacity(high.end - high.start + low.end - low.start);
        // Remove in such order so to retain correct ranges. Higher range should be removed first,
        // otherwise the lower range removal would shift the higher range.
        edges.extend(self.edges.drain(high));
        edges.extend(self.edges.drain(low));

        // Remove node itself.
        let idx = self
            .nodes
            .binary_search_by_key(&id, |n| n.id)
            .ok()
            .expect("node should exist as above code checks for that");
        let node = self.nodes.remove(idx);

        // Unmark the node as a root node.
        self.unroot_node(id);
        
        self.valid.remove_node(id);

        Ok((node, edges))
    }

    pub fn edges(&self) -> impl Iterator<Item = Edge> + '_ {
        self.edges.iter().copied()
    }

    pub(crate) fn edges_inner(&self) -> impl Iterator<Item = EdgeInner> + '_ {
        self.edges.iter().map(|e| EdgeInner {
            from: (self.node_id_to_idx(e.from.node_id).unwrap(), e.from.order),
            to: (self.node_id_to_idx(e.to.node_id).unwrap(), e.to.order),
        })
    }

    /// Add the edge to the canvas. This will return the index of the edge in the canvas.
    /// If the edge already exists, the existing index is returned.
    /// If the nodes referenced by the edge do not exist, this function will return an error.
    ///
    /// This does not validate pin numbers. The validation for that is performed later on
    /// validation/resolution step.
    // NOTE: insertion performance is low for old nodes due to inserts onto the array beginning.
    // Consider optimizing this if it becomes a bottleneck.
    pub fn add_edge_by_parts(
        &mut self,
        from: OutputPin,
        to: InputPin,
    ) -> Result<Edge, NodeNotFoundError> {
        self.node(from.node_id)
            .ok_or(NodeNotFoundError(from.node_id))?;
        self.node(to.node_id).ok_or(NodeNotFoundError(to.node_id))?;
        self.unroot_node(to.node_id);

        let edge = Edge { from, to };
        match self.edges.binary_search(&edge) {
            Ok(_) => {
                // Already added.
            }
            Err(idx) => {
                self.edges.insert(idx, edge);
            }
        };

        Ok(edge)
    }

    /// See [Canvas::add_edge_by_parts].
    pub fn add_edge(&mut self, edge: Edge) -> Result<(), NodeNotFoundError> {
        self.add_edge_by_parts(edge.from, edge.to)?;
        self.valid.add_edge(edge);
        Ok(())
    }

    /// Remove the edge from the canvas.
    pub fn remove_edge(&mut self, edge: Edge) -> Result<(), EdgeNotFoundError> {
        let idx = self
            .edges
            .binary_search(&edge)
            .map_err(|_| EdgeNotFoundError(edge))?;
        self.edges.remove(idx);
        self.valid.remove_edge(edge);
        Ok(())
    }

    /// Remove the root node from the list of root nodes, if it is present.
    fn unroot_node(&mut self, node: Id) {
        if let Some(idx) = self.root_nodes.iter().position(|&n| n == node) {
            self.root_nodes.swap_remove(idx);
        }
    }

    /// Get predicate by its ID.
    pub fn predicate(&self, id: Id) -> Option<&PredicateImpl<NodeMeta>> {
        self.predicates
            .binary_search_by(|p| p.id.cmp(&id))
            .ok()
            .map(|idx| &self.predicates[idx].imp)
    }

    /// Add more input pins to the node.
    /// Moving existing edges accordingly, if necessary.
    /// Range defines the range of pins to add. Range should be within the existing input
    /// pins range of the node, or just after the last pin.
    pub fn add_node_input(
        &mut self,
        node_id: Id,
        range: std::ops::Range<PinOrder>,
    ) -> Result<(), ChangePinCountError> {
        let node = self.node_mut(node_id).ok_or(NodeNotFoundError(node_id))?;

        trace!("calculate extra pin count to add to node {node_id} for added pin range {range:?}");
        let new_count = {
            let current = node.stub.input_pin_count();
            let start = range.start;

            if start > current {
                return Err(ChangePinCountError::InvalidRange(range));
            } else {
                range.len() as PinOrder - (current - start)
            }
        };

        if let Err(e) = node.stub.change_input_count(new_count) {
            warn!("failed to add pins to the node {node_id}. {e}");
            return Err(e.into());
        }

        trace!("shift edges to accommodate added pins");
        let shift_at = range.start;
        let shift_cnt = range.len() as PinOrder;
        let (inp, _) = self
            .node_edge_io_slices_mut(node_id)
            .expect("we found the node in this call already");
        for edge in inp {
            if edge.to.order >= shift_at {
                edge.to.order += shift_cnt;
            }
        }

        Ok(())
    }

    /// Remove node input pins. This is the opposite of [Canvas::add_node_input].
    /// Deleting and moving existing edges accordingly, if necessary.
    pub fn remove_node_input(
        &mut self,
        node_id: Id,
        range: std::ops::Range<PinOrder>,
    ) -> Result<(), ChangePinCountError> {
        let node = self.node_mut(node_id).ok_or(NodeNotFoundError(node_id))?;

        trace!("calculate extra pin count to remove from node {node_id} for removed pin range {range:?}");
        let new_count = {
            let current = node.stub.input_pin_count();
            let start = range.start;

            if start > current {
                return Err(ChangePinCountError::InvalidRange(range));
            } else {
                current - range.len() as PinOrder
            }
        };

        if let Err(e) = node.stub.change_input_count(new_count) {
            warn!("failed to remove pins from the node {node_id}. {e}");
            return Err(e.into());
        }

        trace!("remove edges that are no longer valid");
        let mut vec = SmallVec::<[EdgeIdx; 128]>::new();
        let (inp, _) = self
            .node_edge_io_ranges(node_id)
            .expect("we found the node in this call already");
        for edge_idx in inp {
            let edge = &self.edges[edge_idx as usize];
            if range.contains(&edge.to.order) {
                vec.push(edge_idx as EdgeIdx);
            }
        }
        // Remove from the end as this slightly reduces amount of data moves required,
        // due to a usage of simple vector for edge storage.
        for edge_idx in vec.into_iter().rev() {
            self.edges.remove(edge_idx as usize);
        }

        trace!("shift edges to accommodate removed pins");
        let shift_at = range.end;
        let shift_cnt = range.len() as PinOrder;
        let (inp, _) = self
            .node_edge_io_slices_mut(node_id)
            .expect("we found the node in this call already");
        for edge in inp {
            if edge.to.order >= shift_at {
                edge.to.order -= shift_cnt;
            }
        }

        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum ChangePinCountError {
    #[error("failed to append pin to the node. {0}")]
    NodeNotFound(NodeNotFoundError),

    #[error("unapplicable pin appending operation as per node type")]
    Unapplicable,

    #[error("invalid pin range. {0:?}")]
    InvalidRange(std::ops::Range<PinOrder>),
}

impl From<NodeNotFoundError> for ChangePinCountError {
    fn from(e: NodeNotFoundError) -> Self {
        ChangePinCountError::NodeNotFound(e)
    }
}

impl From<InputUneditableError> for ChangePinCountError {
    fn from(_: InputUneditableError) -> Self {
        ChangePinCountError::Unapplicable
    }
}

#[derive(Debug, Error)]
#[error("node `{0}` not found")]
pub struct NodeNotFoundError(pub Id);

#[derive(Debug, Error)]
#[error("edge `{0}` not found")]
pub struct EdgeNotFoundError(pub Edge);

#[derive(Debug)]
pub struct Node<Meta> {
    pub id: Id,
    pub stub: NodeStub,
    pub meta: Meta,
}

pub type PinOrder = u8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Pin {
    /// The ID of the node this pin belongs to.
    pub node_id: Id,

    /// The index of this pin in the node's pin list.
    pub order: PinOrder,
}

#[cfg(test)]
impl Pin {
    /// Construct a pin with only the node ID set, and order set to zero.
    pub(crate) const fn only_node_id(node_id: Id) -> Pin {
        Pin { node_id, order: 0 }
    }

    pub(crate) const fn new(node_id: Id, order: PinOrder) -> Pin {
        Pin { node_id, order }
    }
}

impl Display for Pin {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.node_id, self.order)
    }
}

/// The [Pin] that is used as an ontput from the node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OutputPin(pub Pin);

impl Display for OutputPin {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// The [Pin] that is used as an input to the node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InputPin(pub Pin);

impl Display for InputPin {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Deref for OutputPin {
    type Target = Pin;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for OutputPin {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Deref for InputPin {
    type Target = Pin;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for InputPin {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct EdgeInner {
    pub from: (NodeIdx, PinOrder),
    pub to: (NodeIdx, PinOrder),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Edge {
    pub from: OutputPin,
    pub to: InputPin,
}

impl Display for Edge {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} -> {}", self.from, self.to)
    }
}

/// A unique identifier for a node or edge.
/// [Edge] and [Node] IDs are unique in the entire [Canvas].
/// They are used to refer to specific nodes and edges in the canvas.
///
/// IDs are assigned randomly in corresponding value spaces.
/// Node and edge IDs are assigned from separate ID spaces so that
/// they don't overlap. This guarantees that user won't accidentally
/// refer to a node as an edge or vice versa, which aids preventing some
/// classes of bugs. Random generation is used to prevent accidental
/// ID reuse, which could lead to subtle bugs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Id(IdInnerType);

impl std::fmt::Display for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use IdKind::*;
        match self.id_kind() {
            Node => write!(f, "Node{}", self.unprefix()),
            Predicate => write!(f, "Predicate{}", self.unprefix()),
        }
    }
}

impl Id {
    pub const NODE_PREFIX: IdInnerType = 0x4000_0000;
    pub const PREDICATE_PREFIX: IdInnerType = 0x2000_0000;

    pub const RND_MAX_STEP: IdInnerType = 128;

    /// Allocate a new node ID.
    pub fn new_node_after(id: Id, rng: &mut SmallRng) -> Option<Id> {
        Self::new(id, rng, Self::NODE_PREFIX)
    }

    fn new(id: Id, rng: &mut SmallRng, prefix: IdInnerType) -> Option<Id> {
        use rand::RngCore;
        let step = 1 + rng.next_u32() % Self::RND_MAX_STEP;
        let new_id = id.unprefix().checked_add(step)?;
        if Self::overlaps_prefix(new_id) {
            warn!("generation of new ID failed due to prefix overlap of a new value `{new_id}`");
            None
        } else {
            Some(Id(new_id | prefix))
        }
    }

    fn overlaps_prefix(val: IdInnerType) -> bool {
        val & (Self::NODE_PREFIX | Self::PREDICATE_PREFIX) != 0
    }

    /// Remove all prefix bits from the ID.
    pub const fn unprefix(self) -> IdInnerType {
        self.0 & !Self::NODE_PREFIX & !Self::PREDICATE_PREFIX
    }

    pub const fn get(&self) -> IdInnerType {
        self.0
    }

    /// Kind of the ID.
    ///
    /// # Panic
    /// This function will panic if the ID is neither a node nor a predicate.
    /// This should not be possible unless the ID was constructed manually.
    pub const fn id_kind(&self) -> IdKind {
        let is_node = self.0 & Self::NODE_PREFIX != 0;
        let is_predicate = self.0 & Self::PREDICATE_PREFIX != 0;

        if is_node == is_predicate {
            panic!("ID has to be either a node or a predicate");
        } else if is_node {
            IdKind::Node
        } else if is_predicate {
            IdKind::Predicate
        } else {
            panic!("should not be reachable");
        }
    }
}

/// The kind of the ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IdKind {
    Node,
    Predicate,
}

#[derive(Debug, Clone)]
pub enum NodeStub {
    /// A file that is accepted as the entry of the process.
    /// Project currently supports one file as an input,
    /// but this can change in the future.
    /// This allows to configure some specific parameters when applicable
    /// over the file node. So that we don’t need special UI just
    /// for such configuration. E.g. whether the file is CSV or XLSX...
    File,

    /// Input is a File. Output is an array of records.
    /// User should describe a regular expressions to use to take
    /// a set of columns and represent their values as separate
    /// rows with other data being carried over.
    /// E.g. if file contains columns like “Dependent 1”, “Dependent 2” and “Dep Birth Date 1”,
    /// “Dep Birth Date 2”… user can set up expressions for them and have
    /// each dependent record being output as a separate row. This effectively
    /// allows to “split by” columns and run flow for these new generated rows.
    /// All columns that are not matched by the regex are copied over to each new row.
    ///
    /// Example: `^((Dependent [0-9]+)|(Dep Birth Date [0-9]+))$`.
    SplitBy { regex: Regex },

    /// Input node describes a column that is used for reading the
    /// data of a single row in a table. One node can allow to
    /// accept several different variants of a column name. That in turn allows
    /// to accept several different file formats that share the same structure
    /// and logic. It also this way may accept editions of the same format,
    /// where column name changes, to allow to be backward-compatible with
    /// previous names as well. However, this requires for the file to
    /// only contain just one column name from the set of variants.
    /// Otherwise, validation error should occur (”ambiguous column variants”).
    Input {
        valid_names: SmallVec<[CompactString; 1]>,
    },

    /// This allows to drop the data.
    /// In parser, all data flows should end either by
    /// Drop or Output nodes. Otherwise, error “unfinished data flow”.
    /// This is to ensure that user hadn’t forgotten to match all the data.
    /// We require to explicitly tell to drop the data if that data is not needed.
    Drop,

    /// Output the data into some external process.
    /// Output has a name by which it is identified in the external process.
    Output { ident: CompactString },

    /// Accepts String. Make all ASCII letters lowercase or uppercase (or other such operation).
    StrOp(StrOp),

    /// Check elements to be (not) equal. Input and Output types must allow equality comparison.
    Compare {
        /// Whether to compare on equality (true) or inequality (false).
        eq: bool,
    },

    /// Compare two elements and determine the `Ordering`.
    Ordering,

    /// Parse a String using regular expression.
    /// Regex is statically verified and returns a Result with Record
    /// that contains all named / numbered groups. This can later be used
    /// by “Record Value” node. Effectively can also be used as a value validator.
    Regex(Regex),

    /// Use predicate to find some record(s) in the record set in the table.
    /// Predicate accepts a File, but does not perform exhaustiveness check
    /// (not required for all columns from the file to be defined). This,
    /// for example, can be used to find a matching employee record from the
    /// dependent record. Or all dependent records for one employee.
    /// Returns Result of array of records.
    FindRecord(PredicateRef),

    /// Map values of one type to another values of possibly other type. Maps should be exhaustive.
    Map {
        /// Array of patterns to match input tuples, and corresponding output tuples.
        tuples: SmallVec<[(Pat, Value); 1]>,
        wildcard: Option<Value>,
    },

    /// List of values of some type. Can be used for filtering or
    /// values validation (e.g. to find invalid/unexpected values).
    /// List can also accept tuples, and then this is a list of valid tuples.
    /// In such cases, input pins represent each tuple element. Number of inputs and
    /// outputs should be the same.
    ///
    /// For each item, values should be of the same type.
    List {
        /// List should have at least one value.
        values: SmallVec<[Value; 1]>,
    },

    /// Check the boolean predicate and execute either branch.
    IfElse {
        /// Data inputs number. This effectively also is the number of outputs times two, for
        /// each branch. This node also has one more extra pin (comming first) for the
        /// boolean value to check - it is not counted in this number.
        inputs: PinOrder,
    },

    /// Check Result and execute either branch passing the corresponding value.
    OkOrErr,

    /// Validate the input values with the predicate.
    /// Produces outputs with Result types. Output count is the same as input count, matching
    /// each input with the corresponding output pin.
    /// Each Result variant contains the input value unchanged,
    /// to allow to pass the value forward in the flow with this additional indication
    /// of correctness.
    Validate {
        /// Predicate to validate the values with. It has the same inputs as this node.
        /// Predicate should return true if the values are valid, and false otherwise.
        predicate: PredicateRef,
    },

    /// Select the first value that is not None in the inputs.
    /// If all inputs are None, the output is None.
    /// Input type can be non-optional type, which will be treated as Some value,
    /// and in turn can act as a default value if placed first and all other inputs are None.
    /// If there is a non-Option input, output type is also non-Option, as it is guaranteed
    /// that there is a value to output.
    ///
    /// This node can be used to receive values produced by [NodeStub::IfElse] or similar
    /// nodes, to merge the branches back into one flow.
    SelectFirst { inputs: PinOrder },

    /// Expect only one input to be Some, and return the value as Ok value, unwrapping
    /// from Option. Otherwise, return Err with the array of all values, unwrapped from Option.
    /// This array will be empty if all inputs are None.
    ExpectOne { inputs: PinOrder },

    /// Expect optional input to be Some, and return the value. Otherwise, crash with the
    /// provided message. Output type is unwrapped from the Option.
    ExpectSome {
        /// Message to display when crashing, can be empty (then ignored).
        msg: CompactString,
    },

    /// Match the input values and execute appropriate branch.
    /// Match should be exhaustive (or with wildcard).
    ///
    /// If match accepts several inputs, these are merged into the array
    /// (since we have no tuple types yet).
    /// Similarly, if output value is an array, it is split into separate outputs.
    Match {
        /// Number of inputs of the node.
        inputs: PinOrder,

        values: SmallVec<[(Pat, Value); 1]>,
        wildcard: Option<Value>,
    },

    /// Custom code function with any amount of input and output pins.
    /// Such node returns corresponding Predicate type if itself is passed as an argument,
    /// or accepts-returns corresponding values on direct flow use.
    /// Whether this is a flow function call or predicate value is defined
    /// in node instance config panel.
    ///
    /// Predicates are useful if we are writing library project and want to export
    /// these function to be used elsewhere.
    /// Flow execution functions are needed to change or compute the values.
    Func(PredicateRef),

    /// Parse the date of the given format(s).
    /// Input is a String and output is Result of a Date,
    /// Time or DateTime. Error is returned if failed.
    ParseDateTime {
        format: CompactString,
        date: bool,
        time: bool,
    },

    /// Parse monetary values of a given format. Input is a String,
    /// output is Monetary value or Error.
    ParseMonetary {
        // Parse with given regex, which should return groups: sign, full dollars, fraction part.
        regex: Regex,
    },

    /// Parse (unsigned) integer from a String. Output is Result.
    ParseInt { signed: bool },

    /// A constant value of some type.
    Constant(Value),

    /// Get Ok value of Result and proceed, or crash on Error variant with optional message.
    OkOrCrash {
        /// Message to display when crashing, can be empty (then ignored).
        msg: CompactString,
    },

    /// Crash the flow with the provided message, if it reaches this node during execution.
    Crash {
        /// Message to display when crashing, can be empty (then ignored).
        msg: CompactString,
    },

    /// The same as [NodeStub::Crash], but with different semantics.
    /// To be used when some flow is believed to never reach this node.
    Unreachable {
        /// Message to display when crashing, can be empty (then ignored).
        msg: CompactString,
    },

    /// To mark unimplemented flow that should be eventually implemented.
    /// To allow saving the project that is WIP as to pass exhaustiveness
    /// (and other) validations.
    /// Project with outstanding Todo nodes cannot be "deployed".
    ///
    /// Todo can have many inputs and outputs, which just directly pass the value(s) forward.
    Todo {
        /// Message to display when crashing, can be empty (then ignored).
        msg: CompactString,

        /// Number of inputs of the node. This effectively also is the number of outputs.
        inputs: PinOrder,
    },

    /// Just some comment, for humans. Can have many inputs and outputs,
    /// which just directly pass the value(s) forward. Must have the same number of ins and outs.
    /// It functions the same as [NodeStub::Todo], but allows to pass all validations
    /// and deployment.
    Comment {
        msg: CompactString,

        /// Number of inputs of the node. This effectively also is the number of outputs.
        inputs: PinOrder,
    },
}

impl NodeStub {
    /// Output pins of the node, if can be determined statically.
    /// Returns `None` if the outputs cannot be determined statically.
    /// Returned slice contains `None` for each output pin that cannot be determined statically,
    /// otherwise contains `Some` with the type of the output pin.
    pub const fn static_outputs(&self) -> Option<&'static [PrimitiveTypeConst]> {
        use NodeStub::*;
        use PrimitiveTypeConst as PT;
        let some: &[PT] = match self {
            File => &[PT::File],
            SplitBy { .. } => &[PT::Array(&PT::Record)],
            Input { .. } => &[PT::Str],
            Drop => &[],
            Output { .. } => &[],
            StrOp(_) => &[PT::Str],
            Compare { .. } => &[PT::Bool],
            Ordering => &[PT::Ordering],
            Regex(_) => return None,
            FindRecord(_) => &[PT::Result((&(PT::Array(&PT::Record)), &PT::Unit))],
            Map { .. } => return None,
            List { .. } => return None,
            IfElse { .. } => return None,
            OkOrErr => &[PT::Hint, PT::Hint],
            Validate { .. } => return None,
            SelectFirst { .. } => &[PT::Hint],
            ExpectOne { .. } => &[PT::Result((&PT::Hint, &PT::Hint))],
            ExpectSome { .. } => &[PT::Hint],
            Match { .. } => return None,
            Func(_) => return None,
            ParseDateTime { date, time, .. } => match (date, time) {
                (true, false) => &[PT::Result((&(PT::Date), &PT::Unit))],
                (false, true) => &[PT::Result((&PT::Time, &PT::Unit))],
                (true, true) => &[PT::Result((&PT::DateTime, &PT::Unit))],
                (false, false) => &[PT::Result((&PT::Unit, &PT::Unit))],
            },
            ParseMonetary { .. } => &[PT::Result((&PT::Moneraty, &PT::Unit))],
            ParseInt { .. } => &[PT::Result((&PT::Int, &PT::Unit))],
            Constant(_) => &[PT::Hint],
            Crash { .. } => &[],
            OkOrCrash { .. } => &[PT::Hint],
            Unreachable { .. } => &[],
            Todo { .. } => return None,
            Comment { .. } => return None,
        };
        Some(some)
    }

    /// How input and output pins relate to each other typewise. When
    /// some input type should be the same as some output type, this function
    /// returns an array of tuples, where each tuple contains the index of the input
    /// and the index of the output pin that should be the same.
    pub const fn static_io_relation(&self) -> IoRelation {
        use IoRelation::*;
        use NodeStub::*;

        let v = match self {
            File => Defined,
            SplitBy { .. } => Dynamic,
            Input { .. } => Defined,
            Drop => ConstCount,
            Output { .. } => ConstCount,
            StrOp(_) => FullSymmetry,
            Compare { .. } => Same(&[(0, 1)]),
            Ordering => Same(&[(0, 1)]),
            Regex(_) => Dynamic,
            FindRecord(_) => Dynamic,
            Map { .. } => Dynamic,
            List { .. } => FullSymmetry,
            IfElse { .. } => Dynamic,
            OkOrErr => Same(&[(1, 2)]),
            Validate { .. } => Dynamic,
            SelectFirst { .. } => Dynamic,
            ExpectOne { .. } => Dynamic,
            ExpectSome { .. } => ConstCount,
            Match { .. } => Dynamic,
            Func(_) => Dynamic,
            ParseDateTime { .. } => Defined,
            ParseMonetary { .. } => Defined,
            ParseInt { .. } => Defined,
            Constant(_) => ConstCount,
            Crash { .. } => ConstCount,
            OkOrCrash { .. } => ConstCount,
            Unreachable { .. } => ConstCount,
            Todo { .. } => FullSymmetry,
            Comment { .. } => FullSymmetry,
        };

        if matches!(v, IoRelation::ConstCount) {
            let is_valid = self.static_inputs().is_some() && self.static_outputs().is_some();
            if !is_valid {
                panic!("ConstCount relation requires known static input and output counts");
            }
        }

        v
    }

    /// Input pins of the node, if can be determined statically.
    /// Returns `None` if the inputs cannot be determined statically.
    /// Returned slice contains `None` for each input pin that cannot be determined statically,
    /// otherwise contains `Some` with the type of the input pin.
    pub const fn static_inputs(&self) -> Option<&'static [PrimitiveTypeConst]> {
        use NodeStub::*;
        use PrimitiveTypeConst as PT;
        let some: &[PT] = match self {
            File => &[],
            SplitBy { .. } => &[PT::Record],
            Input { .. } => &[PT::File],
            Drop => &[PT::Hint],
            Output { .. } => &[PT::Hint],
            StrOp(_) => &[PT::Str],
            Compare { .. } => &[PT::Hint, PT::Hint],
            Ordering => &[PT::Hint, PT::Hint],
            Regex(_) => &[PT::Str],
            FindRecord(_) => &[PT::File],
            Map { .. } => return None,
            List { .. } => return None,
            IfElse { .. } => return None,
            OkOrErr => &[PT::Hint],
            Validate { .. } => return None,
            SelectFirst { .. } => return None,
            ExpectOne { .. } => return None,
            ExpectSome { .. } => &[PT::Option(&PT::Hint)],
            Match { .. } => return None,
            Func(_) => return None,
            ParseDateTime { .. } => &[PT::Str],
            ParseMonetary { .. } => &[PT::Str],
            ParseInt { .. } => &[PT::Str],
            Constant(_) => &[],
            Crash { .. } => &[PT::Hint],
            OkOrCrash { .. } => &[PT::Result((&PT::Hint, &PT::Hint))],
            Unreachable { .. } => &[PT::Hint],
            Todo { .. } => return None,
            Comment { .. } => return None,
        };
        Some(some)
    }

    /// Change the input count if the node has a flexible number of input pins.
    /// Error is returned if the node does not support changing the input count.
    pub const fn change_input_count(&mut self, new: PinOrder) -> Result<(), InputUneditableError> {
        use NodeStub::*;
        match self {
            File
            | SplitBy { .. }
            | Input { .. }
            | Drop
            | Output { .. }
            | StrOp(_)
            | Compare { .. }
            | Ordering
            | Regex(_)
            | FindRecord(_)
            | Map { .. }
            | List { .. }
            | OkOrErr
            | ExpectSome { .. }
            | Validate { .. }
            | Func(_)
            | ParseDateTime { .. }
            | ParseMonetary { .. }
            | ParseInt { .. }
            | Constant(_)
            | Crash { .. }
            | OkOrCrash { .. }
            | Unreachable { .. }
            | Todo { .. }
            | Comment { .. } => return Err(InputUneditableError),

            IfElse { inputs }
            | SelectFirst { inputs }
            | ExpectOne { inputs }
            | Match { inputs, .. } => *inputs = new,
        }

        Ok(())
    }

    pub const fn static_total_pin_count(&self) -> Option<PinOrder> {
        match (self.static_inputs(), self.static_outputs()) {
            (Some(inputs), Some(outputs)) => {
                Some(inputs.len() as PinOrder + outputs.len() as PinOrder)
            }
            _ => None,
        }
    }

    pub fn total_pin_count(&self) -> PinOrder {
        if let Some(v) = self.static_total_pin_count() {
            return v;
        }

        self.input_pin_count() + self.output_pin_count()
    }

    pub fn output_pin_count(&self) -> PinOrder {
        if let Some(v) = self.static_outputs() {
            return v.len() as PinOrder;
        }

        use NodeStub::*;
        match self {
            Comment { .. } | Todo { .. } => self.input_pin_count(),
            Func(predicate) => predicate.output_pin_count(),
            Match { values, .. } => values.len() as PinOrder,
            List { .. } => self.input_pin_count(),
            IfElse { inputs, .. } => *inputs * 2,
            Validate { predicate } => predicate.input_pin_count(), // we pass inputs forward
            Regex(regex) => regex.captures_len() as PinOrder,
            Map { tuples, .. } => tuples
                .first()
                .map(|(_, v)| v.array_len().unwrap_or(1) as PinOrder)
                .unwrap_or(1),
            _ => unreachable!("total_pin_count should be implemented for {self:?}"),
        }
    }

    pub fn input_pin_count(&self) -> PinOrder {
        if let Some(v) = self.static_inputs() {
            return v.len() as PinOrder;
        }

        use NodeStub::*;
        match self {
            Comment { inputs, .. } | Todo { inputs, .. } => *inputs,
            Func(predicate) => predicate.input_pin_count(),
            Match { inputs, .. } => *inputs,
            List { values } => values[0].array_len().unwrap_or(1) as PinOrder,
            IfElse { inputs } => *inputs + 1,
            Validate { predicate } => predicate.input_pin_count(),
            SelectFirst { inputs } => *inputs,
            ExpectOne { inputs } => *inputs,
            Map { .. } => self.output_pin_count(),
            _ => unreachable!("input_pin_count should be implemented for {self:?}"),
        }
    }

    /// Index at which the output pins start counting.
    pub fn output_pin_start_idx(&self) -> PinOrder {
        self.input_pin_count()
    }

    pub fn is_valid_output_ordinal(&self, ordinal: PinOrder) -> bool {
        ordinal < (self.output_pin_start_idx() + self.output_pin_count())
            && ordinal >= self.output_pin_start_idx()
    }

    pub fn is_valid_input_ordinal(&self, ordinal: PinOrder) -> bool {
        ordinal < self.input_pin_count()
    }

    /// Calculate the real pin index in the array of all pins in the
    /// node for the given output pin index. This function accounts
    /// for correct offset of the pins to produce the correct index
    /// to use to refer to this pin.
    ///
    /// # Safety
    /// This does not check whether this pin actually exists, and if it does not
    /// it may overlap with other pins or cause other undefined behavior.
    pub unsafe fn real_output_pin_idx_unchecked(&self, idx: PinOrder) -> PinOrder {
        idx as PinOrder + self.output_pin_start_idx()
    }

    /// See [Self::real_output_pin_idx_unchecked], but this returns [None] on invalid index.
    pub fn real_output_pin_idx(&self, idx: PinOrder) -> Option<PinOrder> {
        if idx >= self.output_pin_start_idx() + self.output_pin_count() {
            None
        } else {
            Some(unsafe { self.real_output_pin_idx_unchecked(idx) })
        }
    }

    /// Calculate the real pin index in the array of all pins in the
    /// node for the given input pin index. This function accounts
    /// for correct offset of the pins to produce the correct index
    /// to use to refer to this pin.
    ///
    /// # Safety
    /// This does not check whether this pin actually exists, and if it does not
    /// it may overlap with other pins or cause other undefined behavior.
    pub unsafe fn real_input_pin_idx_unchecked(&self, idx: PinOrder) -> PinOrder {
        idx as PinOrder
    }

    /// See [Self::real_input_pin_idx_unchecked], but this returns [None] on invalid index.
    pub fn real_input_pin_idx(&self, idx: PinOrder) -> Option<PinOrder> {
        if idx >= self.input_pin_count() {
            None
        } else {
            Some(unsafe { self.real_input_pin_idx_unchecked(idx) })
        }
    }

    /// Whether the node can be used as a starting point of a flow.
    pub const fn is_start(&self) -> bool {
        matches!(self, NodeStub::File)
    }

    pub const fn is_predicate(&self) -> bool {
        use NodeStub::*;
        matches!(self, Func(_) | Validate { .. } | FindRecord(_))
    }
}

#[derive(Debug, Error)]
#[error("input pin count cannot be changed per node stub kind")]
pub struct InputUneditableError;

pub enum IoRelation {
    /// These pin numbers should be the same type.
    Same(&'static [(PinOrder, PinOrder)]),

    /// All input pins have matching output pins.
    FullSymmetry,

    /// Pins have dynamic relation and/or number.
    Dynamic,

    /// Pins have constant count and probably some defined types or type hints.
    ConstCount,

    /// All pin types are fully determined in the constant scope.
    Defined,
}

/// Pattern for matching values.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pat {
    value: Value,
}

impl Pat {
    pub fn from_variants(variants: Vec<Value>) -> Self {
        Self {
            value: Value::Array(variants.into()),
        }
    }

    pub fn from_value(value: Value) -> Self {
        Self { value }
    }

    /// Get the variants of the pattern, if it is an "or" pattern.
    pub fn as_variants(&self) -> Option<&[Value]> {
        match self.value {
            Value::Array(ref v) => Some(v),
            _ => None,
        }
    }

    /// Whether these two patterns can be inside the same pattern list.
    /// Basically, this checks type compatibility.
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.value.type_of() == other.value.type_of()
    }

    pub fn value(&self) -> &Value {
        &self.value
    }

    pub fn into_value(self) -> Value {
        self.value
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Value {
    Int(i64),
    Uint(u64),
    Unit,
    Moneraty(BigDecimal),
    Date(chrono::NaiveDate),
    DateTime(chrono::NaiveDateTime),
    Time(chrono::NaiveTime),
    Bool(bool),
    Str(CompactString),
    Ordering(std::cmp::Ordering),
    Array(Vec<Value>),
    Result {
        value: Result<Box<Value>, Box<Value>>,

        /// A value of the other variant than the one that is present.
        /// This is used to determine the type of the other variant.
        ///
        /// We store just one to reduce [Value] size.
        other: PrimitiveType,
    },
    Option {
        value: Option<Box<Value>>,
        some: PrimitiveType,
    },
}

impl Value {
    pub fn type_of(&self) -> PrimitiveType {
        use Value::*;
        match self {
            Int(_) => PrimitiveType::Int,
            Uint(_) => PrimitiveType::Uint,
            Unit => PrimitiveType::Unit,
            Moneraty(_) => PrimitiveType::Moneraty,
            Date(_) => PrimitiveType::Date,
            DateTime(_) => PrimitiveType::DateTime,
            Time(_) => PrimitiveType::Time,
            Bool(_) => PrimitiveType::Bool,
            Str(_) => PrimitiveType::Str,
            Ordering(_) => PrimitiveType::Ordering,
            Array(v) => PrimitiveType::Array(Box::new(v[0].type_of())),
            Result { value, other } => match value {
                Ok(v) => PrimitiveType::Result(Box::new((v.type_of(), other.clone()))),
                Err(v) => PrimitiveType::Result(Box::new((other.clone(), v.type_of()))),
            },
            Option { some, .. } => PrimitiveType::Option(Box::new(some.to_owned())),
        }
    }

    /// Get the length of the array, if this value is an array.
    pub fn array_len(&self) -> Option<usize> {
        match self {
            Value::Array(v) => Some(v.len()),
            _ => None,
        }
    }

    /// Whether this value has the same type as the other value.
    pub fn is_same_type(&self, other: &Self) -> bool {
        self.type_of() == other.type_of()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Value> {
        return match self {
            Value::Array(v) => Iter::Array(v.iter()),
            v => Iter::Single(v),
        };

        enum Iter<'a> {
            Array(std::slice::Iter<'a, Value>),
            Single(&'a Value),
            Exhausted,
        }

        impl<'a> Iterator for Iter<'a> {
            type Item = &'a Value;

            fn next(&mut self) -> Option<Self::Item> {
                if let Iter::Array(iter) = self {
                    iter.next()
                } else {
                    match std::mem::replace(self, Iter::Exhausted) {
                        Iter::Single(v) => Some(v),
                        Iter::Exhausted => None,
                        Iter::Array(_) => unreachable!(),
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrimitiveType {
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
    Array(Box<PrimitiveType>),
    Result(Box<(PrimitiveType, PrimitiveType)>),
    Option(Box<PrimitiveType>),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum HintedPrimitiveType {
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
    Array(Box<HintedPrimitiveType>),
    Result(Box<(HintedPrimitiveType, HintedPrimitiveType)>),
    Option(Box<HintedPrimitiveType>),
    Hint,
}

impl From<PrimitiveType> for HintedPrimitiveType {
    fn from(pt: PrimitiveType) -> Self {
        use PrimitiveType as PT;
        match pt {
            PT::Int => HintedPrimitiveType::Int,
            PT::Uint => HintedPrimitiveType::Uint,
            PT::Unit => HintedPrimitiveType::Unit,
            PT::Moneraty => HintedPrimitiveType::Moneraty,
            PT::Date => HintedPrimitiveType::Date,
            PT::DateTime => HintedPrimitiveType::DateTime,
            PT::Time => HintedPrimitiveType::Time,
            PT::Bool => HintedPrimitiveType::Bool,
            PT::Str => HintedPrimitiveType::Str,
            PT::Ordering => HintedPrimitiveType::Ordering,
            PT::File => HintedPrimitiveType::File,
            PT::Record => HintedPrimitiveType::Record,
            PT::Array(v) => HintedPrimitiveType::Array(Box::new((*v).into())),
            PT::Result(b) => HintedPrimitiveType::Result(Box::new((b.0.into(), b.1.into()))),
            PT::Option(v) => HintedPrimitiveType::Option(Box::new((*v).into())),
        }
    }
}

impl From<&PrimitiveType> for HintedPrimitiveType {
    fn from(pt: &PrimitiveType) -> Self {
        use PrimitiveType as PT;
        match pt {
            PT::Int => HintedPrimitiveType::Int,
            PT::Uint => HintedPrimitiveType::Uint,
            PT::Unit => HintedPrimitiveType::Unit,
            PT::Moneraty => HintedPrimitiveType::Moneraty,
            PT::Date => HintedPrimitiveType::Date,
            PT::DateTime => HintedPrimitiveType::DateTime,
            PT::Time => HintedPrimitiveType::Time,
            PT::Bool => HintedPrimitiveType::Bool,
            PT::Str => HintedPrimitiveType::Str,
            PT::Ordering => HintedPrimitiveType::Ordering,
            PT::File => HintedPrimitiveType::File,
            PT::Record => HintedPrimitiveType::Record,
            PT::Array(v) => HintedPrimitiveType::Array(Box::new((**v).to_owned().into())),
            PT::Result(b) => HintedPrimitiveType::Result(Box::new((
                b.0.to_owned().into(),
                b.1.to_owned().into(),
            ))),
            PT::Option(v) => HintedPrimitiveType::Option(Box::new((**v).to_owned().into())),
        }
    }
}

impl From<HintedPrimitiveType> for PrimitiveType {
    fn from(hpt: HintedPrimitiveType) -> Self {
        use HintedPrimitiveType as HPT;
        match hpt {
            HPT::Int => PrimitiveType::Int,
            HPT::Uint => PrimitiveType::Uint,
            HPT::Unit => PrimitiveType::Unit,
            HPT::Moneraty => PrimitiveType::Moneraty,
            HPT::Date => PrimitiveType::Date,
            HPT::DateTime => PrimitiveType::DateTime,
            HPT::Time => PrimitiveType::Time,
            HPT::Bool => PrimitiveType::Bool,
            HPT::Str => PrimitiveType::Str,
            HPT::Ordering => PrimitiveType::Ordering,
            HPT::File => PrimitiveType::File,
            HPT::Record => PrimitiveType::Record,
            HPT::Array(v) => PrimitiveType::Array(Box::new((*v).into())),
            HPT::Result(b) => PrimitiveType::Result(Box::new((b.0.into(), b.1.into()))),
            HPT::Option(v) => PrimitiveType::Option(Box::new((*v).into())),
            HPT::Hint => {
                panic!("correct type resolver should not pass naked Hint into resolved type")
            }
        }
    }
}

impl From<&HintedPrimitiveType> for PrimitiveType {
    fn from(hpt: &HintedPrimitiveType) -> Self {
        use HintedPrimitiveType as HPT;
        match hpt {
            HPT::Int => PrimitiveType::Int,
            HPT::Uint => PrimitiveType::Uint,
            HPT::Unit => PrimitiveType::Unit,
            HPT::Moneraty => PrimitiveType::Moneraty,
            HPT::Date => PrimitiveType::Date,
            HPT::DateTime => PrimitiveType::DateTime,
            HPT::Time => PrimitiveType::Time,
            HPT::Bool => PrimitiveType::Bool,
            HPT::Str => PrimitiveType::Str,
            HPT::Ordering => PrimitiveType::Ordering,
            HPT::File => PrimitiveType::File,
            HPT::Record => PrimitiveType::Record,
            HPT::Array(v) => PrimitiveType::Array(Box::new((**v).to_owned().into())),
            HPT::Result(b) => {
                PrimitiveType::Result(Box::new((b.0.to_owned().into(), b.1.to_owned().into())))
            }
            HPT::Option(v) => PrimitiveType::Option(Box::new((**v).to_owned().into())),
            HPT::Hint => {
                panic!("correct type resolver should not pass naked Hint into resolved type")
            }
        }
    }
}

impl HintedPrimitiveType {
    /// Whether the types are compatible. This means they should be equal if neither is a hint,
    /// or if any one is a hint, whether the known parts are equal, ignoring unknown parts.
    pub fn is_compatible(&self, with: &Self) -> bool {
        use HintedPrimitiveType as HPT;
        match (self, with) {
            (HPT::Int, HPT::Int)
            | (HPT::Uint, HPT::Uint)
            | (HPT::Unit, HPT::Unit)
            | (HPT::Moneraty, HPT::Moneraty)
            | (HPT::Date, HPT::Date)
            | (HPT::DateTime, HPT::DateTime)
            | (HPT::Time, HPT::Time)
            | (HPT::Bool, HPT::Bool)
            | (HPT::Str, HPT::Str)
            | (HPT::Ordering, HPT::Ordering)
            | (HPT::File, HPT::File)
            | (HPT::Record, HPT::Record)
            | (HPT::Hint, _) => true,
            (HPT::Array(a), HPT::Array(b)) => a.is_compatible(b),
            (HPT::Result(a), HPT::Result(b)) => a.0.is_compatible(&b.0) && a.1.is_compatible(&b.1),
            (HPT::Option(a), HPT::Option(b)) => a.is_compatible(b),
            _ => false,
        }
    }

    /// This type precision compared to another.
    pub(crate) fn precision(&self, other: &Self) -> TypePrecision {
        use HintedPrimitiveType as HPT;
        use TypePrecision::*;

        match (self, other) {
            (HPT::Int, HPT::Int)
            | (HPT::Uint, HPT::Uint)
            | (HPT::Unit, HPT::Unit)
            | (HPT::Moneraty, HPT::Moneraty)
            | (HPT::Date, HPT::Date)
            | (HPT::DateTime, HPT::DateTime)
            | (HPT::Time, HPT::Time)
            | (HPT::Bool, HPT::Bool)
            | (HPT::Str, HPT::Str)
            | (HPT::Ordering, HPT::Ordering)
            | (HPT::File, HPT::File)
            | (HPT::Record, HPT::Record)
            | (HPT::Hint, HPT::Hint) => Same,
            (HPT::Hint, _) => Uncertain,
            (HPT::Array(a), HPT::Array(b)) => a.precision(b),
            (HPT::Result(a), HPT::Result(b)) => {
                let a0 = a.0.precision(&b.0);
                let a1 = a.1.precision(&b.1);
                match (a0, a1) {
                    (Same, Same) => Same,
                    (Incompatible, _) | (_, Incompatible) => Incompatible,
                    (Precise, _) | (_, Precise) => Precise,
                    (Uncertain, _) | (_, Uncertain) => Uncertain,
                }
            }
            (HPT::Option(a), HPT::Option(b)) => a.precision(b),
            _ => Incompatible,
        }
    }

    /// Make a more defined type from two types. This is used to clarify types
    /// when they are compatible, but one is more precise than the other. This replaces
    /// hints with actual types where applicable. If types are equal, unchanged type is returned.
    /// On error, returns original `self` type.
    ///
    /// [TypePrecision] returned indicates whether the type got more precise, did not improve
    /// ([TypePrecision::Same]) or the types are incompatible ([TypePrecision::Incompatible]).
    /// Returns [TypePrecision::Uncertain] if the types are compatible,
    /// but given value is less certain and was useless for improvements.
    pub fn clarify(&self, other: &Self) -> (Self, TypePrecision) {
        use HintedPrimitiveType as HPT;
        use TypePrecision::*;

        match (self, other) {
            (HPT::Int, HPT::Int)
            | (HPT::Uint, HPT::Uint)
            | (HPT::Unit, HPT::Unit)
            | (HPT::Moneraty, HPT::Moneraty)
            | (HPT::Date, HPT::Date)
            | (HPT::DateTime, HPT::DateTime)
            | (HPT::Time, HPT::Time)
            | (HPT::Bool, HPT::Bool)
            | (HPT::Str, HPT::Str)
            | (HPT::Ordering, HPT::Ordering)
            | (HPT::File, HPT::File)
            | (HPT::Record, HPT::Record)
            | (HPT::Hint, HPT::Hint) => (self.clone(), Same),
            (HPT::Hint, _) => (other.clone(), Precise),
            (_, HPT::Hint) => (self.clone(), Uncertain),
            (HPT::Array(a), HPT::Array(b)) => {
                let (clarified, precision) = a.clarify(b);
                (HPT::Array(Box::new(clarified)), precision)
            }
            (HPT::Result(a), HPT::Result(b)) => {
                let (a0, p0) = a.0.clarify(&b.0);
                let (a1, p1) = a.1.clarify(&b.1);
                let p = match (p0, p1) {
                    (Same, Same) => Same,
                    (Incompatible, _) | (_, Incompatible) => Incompatible,
                    (Precise, _) | (_, Precise) => Precise,
                    (Uncertain, _) | (_, Uncertain) => Uncertain,
                };
                (HPT::Result(Box::new((a0, a1))), p)
            }
            (HPT::Option(a), HPT::Option(b)) => {
                let (clarified, precision) = a.clarify(b);
                (HPT::Option(Box::new(clarified)), precision)
            }
            _ => (self.clone(), Incompatible),
        }
    }

    pub fn is_resolved(&self) -> bool {
        use HintedPrimitiveType as HPT;
        match self {
            HPT::Int
            | HPT::Uint
            | HPT::Unit
            | HPT::Moneraty
            | HPT::Date
            | HPT::DateTime
            | HPT::Time
            | HPT::Bool
            | HPT::Str
            | HPT::Ordering
            | HPT::File
            | HPT::Record => true,
            HPT::Array(v) => v.is_resolved(),
            HPT::Result(boxed) => boxed.0.is_resolved() && boxed.1.is_resolved(),
            HPT::Option(v) => v.is_resolved(),
            HPT::Hint => false,
        }
    }
}

/// Precision to compare two types, whether they are compatible, one can clarify the other,
/// or they are incompatible at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum TypePrecision {
    /// The types are incompatible.
    Incompatible,

    /// The type is less precise than the other type.
    Uncertain,

    /// The type is the same.
    Same,

    /// The type is more precise than the other type.
    Precise,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveTypeConst {
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
    Array(&'static PrimitiveTypeConst),
    Result((&'static PrimitiveTypeConst, &'static PrimitiveTypeConst)),
    Option(&'static PrimitiveTypeConst),

    /// Special type to sign a hint for type resolver. It does not convert into an actual
    /// type outside the const scope.
    ///
    /// During conversion via [From] implementation, this is converted into [PrimitiveType::Unit].
    Hint,
}

impl From<PrimitiveTypeConst> for HintedPrimitiveType {
    fn from(pt: PrimitiveTypeConst) -> Self {
        use HintedPrimitiveType as HPT;
        use PrimitiveTypeConst as PT;
        match pt {
            PT::Int => HPT::Int,
            PT::Uint => HPT::Uint,
            PT::Unit => HPT::Unit,
            PT::Moneraty => HPT::Moneraty,
            PT::Date => HPT::Date,
            PT::DateTime => HPT::DateTime,
            PT::Time => HPT::Time,
            PT::Bool => HPT::Bool,
            PT::Str => HPT::Str,
            PT::Ordering => HPT::Ordering,
            PT::File => HPT::File,
            PT::Record => HPT::Record,
            PT::Array(v) => HPT::Array(Box::new((*v).into())),
            PT::Result((a, b)) => HPT::Result(Box::new(((*a).into(), (*b).into()))),
            PT::Option(v) => HPT::Option(Box::new((*v).into())),
            PT::Hint => HPT::Hint,
        }
    }
}

/// Reference to a predicate inside the [Canvas]. This also carries information
/// about pin count for [NodeStub] to be able to output the correct number of pins.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PredicateRef {
    /// Identifier of a predicate inside the [Canvas].
    pub id: Id,

    /// Number of inputs.
    pub inputs: PinOrder,

    /// Number of outputs.
    pub outputs: PinOrder,
}

impl PredicateRef {
    pub fn total_pin_count(&self) -> PinOrder {
        self.input_pin_count() + self.output_pin_count()
    }

    pub fn input_pin_count(&self) -> PinOrder {
        self.inputs
    }

    pub fn output_pin_count(&self) -> PinOrder {
        self.outputs
    }
}

/// Implementation details of the predicate.
///
/// Predicate is effectively some function that can take and output
/// some values. It can be used to filter records, find records, etc.
///
/// As predicates (components), external projects that are applicable can be imported.
/// They are then considered external dependencies of the current project.
/// External projects are qualified if they have at least one pin to be callable
/// from the flow.
#[derive(Debug)]
pub enum PredicateImpl<NodeMeta> {
    /// Predicate is purely external and is not implemented in the current project.
    /// It cannot be run with actual values as this required linking to the external project,
    /// so only declared types are used during validation.
    Extern {
        /// The symbol by which the predicate is linked into the final binary.
        symbol: CompactString,

        /// Pin types of the predicate.
        pins: Vec<PrimitiveType>,

        /// Number of inputs. Used to split the pins array to get inputs and outputs.
        input_count: PinOrder,
    },

    /// Predicate is stored in the current project as a component.
    Component {
        /// Nodes as predicate implementation.
        nodes: Vec<Node<NodeMeta>>,

        /// Edges between nodes of the predicate.
        edges: Vec<Edge>,
    },
}

#[derive(Debug)]
pub(crate) struct CanvasPredicate<NodeMeta> {
    pub(crate) id: Id,
    pub(crate) imp: PredicateImpl<NodeMeta>,
}

impl<NodeMeta> PartialEq for CanvasPredicate<NodeMeta> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<NodeMeta> Eq for CanvasPredicate<NodeMeta> {}

impl<NodeMeta> std::hash::Hash for CanvasPredicate<NodeMeta> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<NodeMeta> PartialOrd for CanvasPredicate<NodeMeta> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl<NodeMeta> Ord for CanvasPredicate<NodeMeta> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

/// Operation to apply to a `String` input.
#[derive(Debug, Clone)]
pub enum StrOp {
    /// Convert the string to ASCII lowercase.
    Lowercase,

    /// Convert the string to ASCII uppercase.
    Uppercase,

    Strip {
        /// Trim whitespace from the beginning of the string.
        trim_whitespace: bool,

        /// Trim whitespace from the end of the string.
        trim_end_whitespace: bool,

        /// Remove given substrings from the input.
        remove: SmallVec<[CompactString; 1]>,
    },
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn add_edge() {
        crate::tests::init();

        let stub = NodeStub::Todo {
            msg: CompactString::from("test"),
            inputs: 1,
        };

        let mut canvas = Canvas::<()>::new();
        let a = canvas.add_node(stub.clone(), ());
        let b = canvas.add_node(stub.clone(), ());
        let c = canvas.add_node(stub, ());

        let outa = OutputPin(Pin {
            node_id: a,
            order: 0,
        });
        let inpb = InputPin(Pin {
            node_id: b,
            order: 0,
        });
        let outb = OutputPin(Pin {
            node_id: b,
            order: 0,
        });
        let inpc = InputPin(Pin {
            node_id: c,
            order: 0,
        });
        let outc = OutputPin(Pin {
            node_id: a,
            order: 0,
        });
        let inpa = InputPin(Pin {
            node_id: a,
            order: 0,
        });

        let ab = canvas.add_edge_by_parts(outa, inpb).unwrap();
        let bc = canvas.add_edge_by_parts(outb, inpc).unwrap();
        let ca = canvas.add_edge_by_parts(outc, inpa).unwrap();

        assert_eq!(ab.from, outa);
        assert_eq!(ab.to, inpb);
        assert_eq!(bc.from, outb);
        assert_eq!(bc.to, inpc);
        assert_eq!(ca.from, outc);
        assert_eq!(ca.to, inpa);
    }

    #[test]
    fn node_edge_io_iter() {
        crate::tests::init();

        let stub = NodeStub::Todo {
            msg: CompactString::from("test"),
            inputs: 1,
        };

        let mut canvas = Canvas::<()>::new();
        let a = canvas.add_node(stub.clone(), ());
        let b = canvas.add_node(stub.clone(), ());
        let c = canvas.add_node(stub, ());

        let outa = OutputPin(Pin {
            node_id: a,
            order: 1,
        });
        let inpb = InputPin(Pin {
            node_id: b,
            order: 0,
        });
        let outb = OutputPin(Pin {
            node_id: b,
            order: 1,
        });
        let inpc = InputPin(Pin {
            node_id: c,
            order: 0,
        });
        let outc = OutputPin(Pin {
            node_id: c,
            order: 1,
        });
        let inpa = InputPin(Pin {
            node_id: a,
            order: 0,
        });

        let ab = canvas.add_edge_by_parts(outa, inpb).unwrap();
        let bc = canvas.add_edge_by_parts(outb, inpc).unwrap();
        let ca = canvas.add_edge_by_parts(outc, inpa).unwrap();

        let mut iter = canvas.node_edge_io_iter(a).unwrap();
        assert_eq!(iter.next(), Some(ab));
        assert_eq!(iter.next(), Some(ca));
        assert_eq!(iter.next(), None);

        let mut iter = canvas.node_edge_io_iter(b).unwrap();
        assert_eq!(iter.next(), Some(ab));
        assert_eq!(iter.next(), Some(bc));
        assert_eq!(iter.next(), None);

        let mut iter = canvas.node_edge_io_iter(c).unwrap();
        assert_eq!(iter.next(), Some(bc));
        assert_eq!(iter.next(), Some(ca));
        assert_eq!(iter.next(), None);
    }
}
