use std::{
    fmt::{Display, Formatter},
    ops::{Deref, RangeInclusive},
};

use bigdecimal::BigDecimal;
use compact_str::CompactString;
use lazy_regex::Regex;
use log::trace;
use rand::rngs::SmallRng;
use smallvec::SmallVec;
use thiserror::Error;

#[derive(Debug)]
pub struct Canvas<NodeMeta> {
    pub(crate) nodes: Vec<Node<NodeMeta>>,
    pub(crate) edges: Vec<Edge>,
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
            root_nodes: Vec::new(),
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

        self.root_nodes.push(id);
        self.nodes.push(Node { id, stub, meta });

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

    fn node_edge_io_ranges(
        &self,
        node_id: Id,
    ) -> Option<(RangeInclusive<usize>, RangeInclusive<usize>)> {
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

            start..=end
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

            start..=end
        };

        Some((range_in, range_out))
    }

    fn node_edge_io_slices(&self, node_id: Id) -> Option<(&[Edge], &[Edge])> {
        let (inp, out) = self.node_edge_io_ranges(node_id)?;
        Some((&self.edges[inp], &self.edges[out]))
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
        let (high, low) = if inp.start() > out.start() {
            (inp, out)
        } else {
            (out, inp)
        };
        let mut edges = Vec::with_capacity(high.end() - high.start() + low.end() - low.start());
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
        self.add_edge_by_parts(edge.from, edge.to).map(|_| ())
    }

    /// Remove the edge from the canvas.
    pub fn remove_edge(&mut self, edge: Edge) -> Result<(), EdgeNotFoundError> {
        let idx = self
            .edges
            .binary_search(&edge)
            .map_err(|_| EdgeNotFoundError(edge))?;
        self.edges.remove(idx);
        Ok(())
    }

    /// Remove the root node from the list of root nodes, if it is present.
    fn unroot_node(&mut self, node: Id) {
        if let Some(idx) = self.root_nodes.iter().position(|&n| n == node) {
            self.root_nodes.swap_remove(idx);
        }
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

impl<Meta> Node<Meta> {
    pub fn is_predicate(&self) -> bool {
        self.stub.is_predicate()
    }

    pub fn is_start(&self) -> bool {
        self.stub.is_start()
    }

    /// Count of input pins of the node, if statically known.
    pub const fn static_input_count(&self) -> Option<usize> {
        if let Some(v) = self.stub.static_inputs() {
            Some(v.len())
        } else {
            None
        }
    }

    /// Count of output pins of the node, if statically known.
    pub const fn static_output_count(&self) -> Option<usize> {
        if let Some(v) = self.stub.static_outputs() {
            Some(v.len())
        } else {
            None
        }
    }
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

impl Deref for InputPin {
    type Target = Pin;

    fn deref(&self) -> &Self::Target {
        &self.0
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
        if self.is_node() {
            write!(f, "Node{}", self.unprefix())
        } else {
            write!(f, "Err{}", self.unprefix())
        }
    }
}

impl Id {
    pub const NODE_PREFIX: u32 = 0x4000_0000;
    pub const RND_MAX_STEP: u32 = 128;
    pub const ZERO: Id = Id(0);

    /// Allocate a new node ID.
    pub fn new_node_after(id: Id, rng: &mut SmallRng) -> Option<Id> {
        Self::new(id, rng, Self::NODE_PREFIX)
    }

    fn new(id: Id, rng: &mut SmallRng, prefix: u32) -> Option<Id> {
        use rand::RngCore;
        let step = 1 + rng.next_u32() % Self::RND_MAX_STEP;
        let new_id = id.unprefix().checked_add(step)?;
        Some(Id(new_id | prefix))
    }

    /// Remove all prefix bits from the ID.
    pub const fn unprefix(self) -> u32 {
        self.0 & !Self::NODE_PREFIX
    }

    pub const fn get(&self) -> u32 {
        self.0
    }

    pub const fn is_node(&self) -> bool {
        self.0 & Self::NODE_PREFIX != 0
    }
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
    FindRecord(Predicate),

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
        /// Inputs number excluding the condition predicate inputs.
        inputs: PinOrder,
        condition: Predicate,
    },

    /// Check Result and execute either branch passing the corresponding value.
    OkOrErr,

    /// Match the input values and execute appropriate branch.
    /// Match should be exhaustive (or with wildcard).
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
    Func(Predicate),

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

    /// Make the program crash, with some (optional) message.
    Crash {
        /// Message to display when crashing, can be empty (then ignored).
        msg: CompactString,
    },

    /// Get Ok value of Result and proceed, or crash on Error variant with optional message.
    OkOrCrash {
        /// Message to display when crashing, can be empty (then ignored).
        msg: CompactString,
    },

    /// The same as Crash, but with different semantics.
    /// To be used when some flow is believed to never reach this node.
    Unreachable {
        /// Message to display when crashing, can be empty (then ignored).
        msg: CompactString,
    },

    /// To mark unimplemented flow that should be eventually implemented.
    /// To allow saving the project that is WIP as to pass exhaustiveness
    /// (or other?) validations.
    /// Project with outstanding Todo nodes cannot be “deployed”.
    ///
    /// Todo can have many inputs and outputs, which just directly pass the value(s) forward.
    Todo {
        /// Message to display when crashing, can be empty (then ignored).
        msg: CompactString,

        /// Number of inputs of the node. This effectively also is the number of outputs.
        inputs: PinOrder,
    },

    /// Just some comment, for visuals. Can have many inputs and outputs,
    /// which just directly pass the value(s) forward. Must have the same number of ins and outs.
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
    pub const fn static_outputs(&self) -> Option<&'static [Option<PrimitiveTypeConst>]> {
        use NodeStub::*;
        use PrimitiveTypeConst as PT;
        let some: &[Option<PT>] = match self {
            File => &[Some(PT::File)],
            SplitBy { .. } => &[Some(PT::Array(&PT::Record))],
            Input { .. } => &[Some(PT::Str)],
            Drop => &[],
            Output { .. } => &[],
            StrOp(_) => &[Some(PT::Str)],
            Compare { .. } => &[Some(PT::Bool)],
            Ordering => &[Some(PT::Ordering)],
            Regex(_) => return None,
            FindRecord(_) => &[Some(PT::Result(&(PT::Array(&PT::Record))))],
            Map { .. } => return None,
            List { .. } => return None,
            IfElse { .. } => return None,
            OkOrErr => &[None, None],
            Match { .. } => return None,
            Func(_) => return None,
            ParseDateTime { date, time, .. } => match (date, time) {
                (true, false) => &[Some(PT::Result(&(PT::Date)))],
                (false, true) => &[Some(PT::Result(&(PT::Time)))],
                (true, true) => &[Some(PT::Result(&(PT::DateTime)))],
                (false, false) => &[Some(PT::Result(&(PT::Unit)))],
            },
            ParseMonetary { .. } => &[Some(PT::Result(&(PT::Moneraty)))],
            ParseInt { .. } => &[Some(PT::Result(&(PT::Int)))],
            Constant(_) => &[None],
            Crash { .. } => &[],
            OkOrCrash { .. } => &[None],
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
        match self {
            File => Unspecified,
            SplitBy { .. } => Unspecified,
            Input { .. } => Unspecified,
            Drop => Unspecified,
            Output { .. } => Unspecified,
            StrOp(_) => FullSymmetry,
            Compare { .. } => Same(&[(0, 1)]),
            Ordering => Same(&[(0, 1)]),
            Regex(_) => Unspecified,
            FindRecord(_) => Unspecified,
            Map { .. } => Unspecified,
            List { .. } => FullSymmetry,
            IfElse { .. } => Unspecified,
            OkOrErr => Same(&[(1, 2)]),
            Match { .. } => Unspecified,
            Func(_) => Unspecified,
            ParseDateTime { .. } => Unspecified,
            ParseMonetary { .. } => Unspecified,
            ParseInt { .. } => Unspecified,
            Constant(_) => Unspecified,
            Crash { .. } => Unspecified,
            OkOrCrash { .. } => FullSymmetry,
            Unreachable { .. } => Unspecified,
            Todo { .. } => FullSymmetry,
            Comment { .. } => FullSymmetry,
        }
    }

    /// Input pins of the node, if can be determined statically.
    /// Returns `None` if the inputs cannot be determined statically.
    /// Returned slice contains `None` for each input pin that cannot be determined statically,
    /// otherwise contains `Some` with the type of the input pin.
    pub const fn static_inputs(&self) -> Option<&'static [Option<PrimitiveTypeConst>]> {
        use NodeStub::*;
        use PrimitiveTypeConst as PT;
        let some: &[Option<PT>] = match self {
            File => &[],
            SplitBy { .. } => &[Some(PT::Record)],
            Input { .. } => &[Some(PT::File)],
            Drop => &[],
            Output { .. } => &[None],
            StrOp(_) => &[Some(PT::Str)],
            Compare { .. } => &[None, None],
            Ordering => &[None, None],
            Regex(_) => &[Some(PT::Str)],
            FindRecord(_) => &[Some(PT::File)],
            Map { .. } => return None,
            List { .. } => return None,
            IfElse { .. } => return None,
            OkOrErr => &[None],
            Match { .. } => return None,
            Func(_) => return None,
            ParseDateTime { .. } => &[Some(PT::Str)],
            ParseMonetary { .. } => &[Some(PT::Str)],
            ParseInt { .. } => &[Some(PT::Str)],
            Constant(_) => &[],
            Crash { .. } => &[None],
            OkOrCrash { .. } => &[None],
            Unreachable { .. } => &[None],
            Todo { .. } => return None,
            Comment { .. } => return None,
        };
        Some(some)
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
            IfElse { inputs, .. } => *inputs,
            Regex(regex) => regex.captures_len() as PinOrder,
            Map { tuples, .. } => tuples[0].1.array_len().unwrap_or(1) as PinOrder,
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
            Func(predicate) => predicate.inputs.len() as PinOrder,
            Match { inputs, .. } => *inputs,
            List { values } => values[0].array_len().unwrap_or(1) as PinOrder,
            IfElse { inputs, condition } => condition.inputs.len() as PinOrder + *inputs,
            Map { tuples, .. } => tuples.len() as PinOrder,
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
        matches!(self, Func(_) | IfElse { .. } | FindRecord(_))
    }
}

pub enum IoRelation {
    /// These pin numbers should be the same type.
    Same(&'static [(PinOrder, PinOrder)]),

    /// All input pins have matching output pins.
    FullSymmetry,

    /// Pins have dynamic relation and/or number, or no relation at all.
    Unspecified,
}

/// Result of resolution of pin types of some node.
pub struct ResolvePinTypes<'a> {
    pins: &'a mut Vec<Option<PrimitiveType>>,
    is_progress: bool,
}

impl<'pins> ResolvePinTypes<'pins> {
    pub(crate) fn pin_io_slices_mut<'a>(
        pins: &'a mut [Option<PrimitiveType>],
        node: &NodeStub,
    ) -> (
        &'a mut [Option<PrimitiveType>],
        &'a mut [Option<PrimitiveType>],
    ) {
        let (inputs, outputs) = pins.split_at_mut(node.input_pin_count() as usize);
        (inputs, outputs)
    }

    /// Whether the last iteration has changed any pin type.
    pub fn is_progress(&self) -> bool {
        self.is_progress
    }

    /// Resolve the pin types of the given node.
    /// The `set` parameter is a vector holding the resolved types of the input and output pins.
    /// Resolver cannot change those but it can use them to determine the types of other pins.
    /// Vector should be the same length as the number of pins of the node.
    ///
    /// `expect_progress_or_complete` should be set to true if the context expects to have
    /// any progress in the resolution. If no progress is made, the function will return an error.
    /// The error is silenced for already fully resolved nodes.
    pub fn resolve(
        node: &NodeStub,
        pins: &'pins mut Vec<Option<PrimitiveType>>,
        expect_progress_or_complete: bool,
    ) -> Result<Self, PinResolutionError> {
        if pins.len() as PinOrder != node.total_pin_count() {
            return Err(PinResolutionError::PinNumberMismatch);
        }

        let mut is_progress = ResolvePinTypes::prefill_with_static(node, pins)?;

        let result = match node.static_io_relation() {
            IoRelation::Same(pairs) => {
                for &(i, o) in pairs {
                    let (i, o) = (i as usize, o as usize);
                    let (i, o) = {
                        // To satisfy the borrow checker, we split slice to guarantee
                        // non-overlapping mutable references.
                        let (a, b) = pins.split_at_mut(i + 1);
                        (&mut a[i], &mut b[o - i - 1])
                    };
                    let any = ResolvePinTypes::any(i, o);
                    is_progress |= ResolvePinTypes::set_to(any, i, o);
                }

                Self { pins, is_progress }
            }
            IoRelation::FullSymmetry => {
                let (ins, outs) = Self::pin_io_slices_mut(pins, node);
                for (i, o) in ins.iter_mut().zip(outs.iter_mut()) {
                    is_progress |= ResolvePinTypes::unite(i, o)?;
                }

                Self { pins, is_progress }
            }
            IoRelation::Unspecified => {
                use NodeStub::*;
                match node {
                    Regex { .. } => {
                        // All should be strings.
                        for pin in pins.iter_mut() {
                            is_progress |=
                                ResolvePinTypes::match_types_write(PrimitiveType::Str, pin)?;
                        }

                        Self { pins, is_progress }
                    }
                    Map { tuples, .. } => {
                        let first = &tuples
                            .first()
                            .expect("Map should have at least one tuple")
                            .1;
                        for (i, val) in first.iter().enumerate() {
                            is_progress |=
                                ResolvePinTypes::match_types_write(val.type_of(), &mut pins[i])?;
                        }

                        // Input types should be already provided externally.
                        Self { pins, is_progress }
                    }
                    IfElse { condition, inputs } => {
                        // Output pins have groups for true and false branch.
                        // Otherwise, each of that group is symmetric to input pins, except for
                        // the first ones that go to the condition predicate.

                        let after_predicate_idx = condition.input_pin_count() as usize;
                        let branch_size = *inputs as usize;
                        let (predicate_pins, rest) = pins.split_at_mut(after_predicate_idx);
                        let (input_pins, rest) = rest.split_at_mut(branch_size);
                        let (true_branch, false_branch) = rest.split_at_mut(branch_size);
                        assert_eq!(input_pins.len(), branch_size);
                        assert_eq!(true_branch.len(), branch_size);
                        assert_eq!(false_branch.len(), branch_size);

                        // Resolve input data pins (exclude predicate).
                        for (i, (t, f)) in true_branch
                            .iter_mut()
                            .zip(false_branch.iter_mut())
                            .enumerate()
                        {
                            if let Some(ty) = input_pins[i].as_ref() {
                                is_progress |= ResolvePinTypes::match_types_write(ty.clone(), t)?;
                                is_progress |= ResolvePinTypes::match_types_write(ty.clone(), f)?;
                            }
                        }

                        // Resolve predicate pins.
                        for (i, ty) in condition.inputs.iter().enumerate() {
                            is_progress |= ResolvePinTypes::match_types_write(
                                ty.to_owned(),
                                &mut predicate_pins[i],
                            )?;
                        }

                        Self { pins, is_progress }
                    }
                    Match { .. } => todo!(),
                    Func { .. } => todo!(),
                    Constant(value) => {
                        let ty = value.type_of();
                        assert_eq!(pins.len(), 1);
                        is_progress |= ResolvePinTypes::match_types_write(ty, &mut pins[0])?;

                        Self { pins, is_progress }
                    }
                    _ => Self { pins, is_progress },
                }
            }
        };

        trace!("Resolved pins: {:#?}", result.pins);
        if expect_progress_or_complete && !result.is_progress {
            result.ensure_resolved()
        } else {
            Ok(result)
        }
    }

    /// Prefill missing types per static information.
    ///
    /// # Panics
    /// Pin count should be correct at this point. If it is not, this function will panic.
    fn prefill_with_static(
        node: &NodeStub,
        pins: &mut Vec<Option<PrimitiveType>>,
    ) -> Result<bool, PinResolutionError> {
        let output_idx = node.output_pin_start_idx();
        let mut is_progress = false;

        trace!("static inputs prefill");
        if let Some(static_inputs) = node.static_inputs() {
            debug_assert_eq!(static_inputs.len(), output_idx as usize);
            for (i, t) in static_inputs.iter().copied().enumerate() {
                if let Some(t) = t.map(Into::into) {
                    if pins[i].is_none() {
                        trace!("prefill input pin {i} with {t:?}");
                        is_progress = true;
                        pins[i] = Some(t);
                    } else if pins[i] != Some(t) {
                        return Err(PinResolutionError::UnionConflict);
                    }
                }
            }
        }

        trace!("static outputs prefill");
        if let Some(static_outputs) = node.static_outputs() {
            for (i, t) in static_outputs.iter().copied().enumerate() {
                let i = i + output_idx as usize;
                if let Some(t) = t.map(Into::into) {
                    if pins[i].is_none() {
                        trace!("prefill output pin {i} with {t:?}");
                        is_progress = true;
                        pins[i] = Some(t);
                    } else if pins[i] != Some(t) {
                        return Err(PinResolutionError::UnionConflict);
                    }
                }
            }
        }

        trace!("static prefill progress = {is_progress}");
        Ok(is_progress)
    }

    /// If either of the pins is `None`, they are unified to the same type.
    /// If both are `Some`, they are validated to be the same type.
    /// If both are None, they are left as None and false is returned.
    /// True is returned if the types were unified by making a change.
    /// False is returned if no change was made.
    fn unite(
        a: &mut Option<PrimitiveType>,
        b: &mut Option<PrimitiveType>,
    ) -> Result<bool, UnionConflict> {
        if let Some(a) = a {
            if let Some(b) = b {
                if a != b {
                    Err(UnionConflict)
                } else {
                    Ok(false)
                }
            } else {
                *b = Some(a.clone());
                Ok(true)
            }
        } else if let Some(b) = b {
            *a = Some(b.clone());
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Set all pins to the given value. Return true if any change was made.
    fn set_to(
        val: Option<PrimitiveType>,
        a: &mut Option<PrimitiveType>,
        b: &mut Option<PrimitiveType>,
    ) -> bool {
        let mut changed = false;
        if a != &val {
            *a = val.clone();
            changed = true;
        }
        if b != &val {
            *b = val;
            changed = true;
        }
        changed
    }

    /// If `b` is `Some`, it should be equal to `a`.
    /// Otherwise, `b` is set to `a`.
    /// True is returned if change was made.
    fn match_types_write(
        a: PrimitiveType,
        b: &mut Option<PrimitiveType>,
    ) -> Result<bool, UnionConflict> {
        if let Some(b) = b {
            if &a != b {
                return Err(UnionConflict);
            }
            Ok(false)
        } else {
            *b = Some(a);
            Ok(true)
        }
    }

    fn any(a: &Option<PrimitiveType>, b: &Option<PrimitiveType>) -> Option<PrimitiveType> {
        match (a, b) {
            (Some(a), _) => Some(a.clone()),
            (_, Some(b)) => Some(b.clone()),
            _ => None,
        }
    }

    /// Check that all pins are resolved.
    fn ensure_resolved(self) -> Result<Self, PinResolutionError> {
        if self.is_resolved() {
            Ok(self)
        } else {
            Err(PinResolutionError::RemainingUnknownPins)
        }
    }

    pub fn is_resolved(&self) -> bool {
        for pin in self.pins.iter() {
            if pin.is_none() {
                return false;
            }
        }
        true
    }
}

struct UnionConflict;

impl From<UnionConflict> for PinResolutionError {
    fn from(_: UnionConflict) -> Self {
        PinResolutionError::UnionConflict
    }
}

#[derive(Debug)]
pub enum PinResolutionError {
    PinNumberMismatch,

    /// Provided types of the pins cannot be resolved due to conflicting requirements.
    UnionConflict,

    /// Cannot resolve all pin types for lacking information.
    RemainingUnknownPins,
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
    Predicate(Predicate),
    Result {
        // Value is propagated only on Ok, but we still always carry it around for type resolution.
        value: Box<Value>,
        is_ok: bool,
    },
    Option {
        // We still carry it around for type resolution and possibly some logs.
        value: Box<Value>,
        is_some: bool,
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
            Predicate(p) => PrimitiveType::Predicate(Box::new(p.clone())),
            Result { value, .. } => PrimitiveType::Result(Box::new(value.type_of())),
            Option { value, .. } => PrimitiveType::Option(Box::new(value.type_of())),
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
    Predicate(Box<Predicate>),
    Result(Box<PrimitiveType>),
    Option(Box<PrimitiveType>),
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
    Predicate(&'static PredicateConst),
    Result(&'static PrimitiveTypeConst),
    Option(&'static PrimitiveTypeConst),
}

impl From<PrimitiveTypeConst> for PrimitiveType {
    fn from(pt: PrimitiveTypeConst) -> Self {
        use PrimitiveTypeConst as PT;
        match pt {
            PT::Int => PrimitiveType::Int,
            PT::Uint => PrimitiveType::Uint,
            PT::Unit => PrimitiveType::Unit,
            PT::Moneraty => PrimitiveType::Moneraty,
            PT::Date => PrimitiveType::Date,
            PT::DateTime => PrimitiveType::DateTime,
            PT::Time => PrimitiveType::Time,
            PT::Bool => PrimitiveType::Bool,
            PT::Str => PrimitiveType::Str,
            PT::Ordering => PrimitiveType::Ordering,
            PT::File => PrimitiveType::File,
            PT::Record => PrimitiveType::Record,
            PT::Array(&v) => PrimitiveType::Array(Box::new(v.into())),
            PT::Predicate(&pc) => PrimitiveType::Predicate(Box::new(pc.into())),
            PT::Result(&a) => PrimitiveType::Result(Box::new(a.into())),
            PT::Option(&v) => PrimitiveType::Option(Box::new(v.into())),
        }
    }
}

/// Predicate is effectively some function that can take and output
/// some values. It can be used to filter records, find records, etc.
///
/// As predicates, external projects that are applicable can be imported.
/// They are then considered external dependencies of the current project.
///
/// To qualify as a predicate, the function must be pure and deterministic.
///
/// External projects are qualified if they have at least one pin for the node
/// they will be represented with. They themselves, as functions, should be pure and deterministic.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Predicate {
    pub inputs: SmallVec<[PrimitiveType; 1]>,
    pub outputs: SmallVec<[PrimitiveType; 1]>,
}

impl Predicate {
    pub fn total_pin_count(&self) -> PinOrder {
        self.input_pin_count() + self.output_pin_count()
    }

    pub fn input_pin_count(&self) -> PinOrder {
        self.inputs.len() as PinOrder
    }

    pub fn output_pin_count(&self) -> PinOrder {
        self.outputs.len() as PinOrder
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PredicateConst {
    pub inputs: &'static [PrimitiveTypeConst],
    pub outputs: &'static [PrimitiveTypeConst],
}

impl From<PredicateConst> for Predicate {
    fn from(pc: PredicateConst) -> Self {
        Self {
            inputs: pc.inputs.iter().copied().map(Into::into).collect(),
            outputs: pc.outputs.iter().copied().map(Into::into).collect(),
        }
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
}
