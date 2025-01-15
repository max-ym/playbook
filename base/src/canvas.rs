use bigdecimal::BigDecimal;
use compact_str::CompactString;
use lazy_regex::Regex;
use rand::rngs::SmallRng;
use smallvec::SmallVec;

#[derive(Debug)]
pub struct Canvas<NodeMeta> {
    nodes: Vec<Node<NodeMeta>>,
    edges: Vec<Edge>,
    rnd: SmallRng,

    /// IDs of the root nodes of the canvas.
    /// From these nodes the execution of the flow starts.
    /// Normally there should be only one, but we still allow to store here
    /// multiple and then will show an error during validation.
    /// 
    /// For projects that act as a library, the root nodes should be absent and instead
    /// the project should have at least one input or output pin exposed.
    root_nodes: SmallVec<[Id; 1]>,
}

impl<NodeMeta> Canvas<NodeMeta> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            root_nodes: SmallVec::new(),
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

        if stub.is_root() {
            self.root_nodes.push(id);
        }

        self.nodes.push(Node { id, stub, meta });

        id
    }

    pub fn get_node(&self, id: Id) -> Option<&Node<NodeMeta>> {
        // Node array is sorted, so we can use binary search.
        debug_assert!(self.nodes.is_sorted_by_key(|n| n.id));
        self.nodes
            .binary_search_by_key(&id, |n| n.id)
            .ok()
            .map(|idx| &self.nodes[idx])
    }

    pub fn remove_node(&mut self, id: Id) -> Option<Node<NodeMeta>> {
        let idx = self.nodes.binary_search_by_key(&id, |n| n.id).ok()?;
        let node = self.nodes.remove(idx);

        // Remove all edges that are connected to the removed node.
        self.edges.retain(|e| e.from.node_id != id && e.to.node_id != id);

        // Remove the node from the root nodes.
        self.root_nodes.retain(|root| *root != id);

        Some(node)
    }
}

#[derive(Debug)]
pub struct Node<Meta> {
    pub id: Id,
    pub stub: NodeStub,
    pub meta: Meta,
}

impl<Meta> Node<Meta> {
    pub fn is_predicate(&self) -> bool {
        matches!(self.stub, NodeStub::Func(_))
    }

    pub fn is_root(&self) -> bool {
        self.stub.is_root()
    }

    /// Count of input pins of the node, if statically known.
    pub const fn static_inputs(&self) -> Option<usize> {
        if let Some(v) = self.stub.static_inputs() {
            Some(v.len())
        } else {
            None
        }
    }

    /// Count of output pins of the node, if statically known.
    pub const fn static_outputs(&self) -> Option<usize> {
        if let Some(v) = self.stub.static_outputs() {
            Some(v.len())
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Pin {
    /// The ID of the node this pin belongs to.
    pub node_id: Id,

    /// The index of this pin in the node's pin list.
    pub order: u8,
}

impl Pin {
    /// Construct a zero pin. It has zeroed node ID and order.
    pub(crate) const fn zero() -> Pin {
        Pin {
            node_id: Id(0),
            order: 0,
        }
    }

    /// Construct a pin with only the node ID set, and order set to zero.
    pub(crate) const fn only_node_id(node_id: Id) -> Pin {
        Pin { node_id, order: 0 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Edge {
    pub from: Pin,
    pub to: Pin,
}

impl Edge {
    /// Construct edge for binary search of the node in sorted array.
    pub(crate) const fn binary_search_from(node_id: Id) -> Self {
        Self {
            from: Pin::only_node_id(node_id),
            to: Pin::zero(),
        }
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
pub struct Id(u32);

impl Id {
    pub const NODE_PREFIX: u32 = 0x4000_0000;
    pub const EDGE_PREFIX: u32 = 0x2000_0000;
    pub const RND_MAX_STEP: u32 = 128;

    /// Allocate a new node ID.
    pub fn new_node_after(id: Id, rng: &mut SmallRng) -> Option<Id> {
        Self::new(id, rng, Self::NODE_PREFIX)
    }

    /// Allocate a new edge ID.
    pub fn new_edge_after(id: Id, rng: &mut SmallRng) -> Option<Id> {
        Self::new(id, rng, Self::EDGE_PREFIX)
    }

    fn new(id: Id, rng: &mut SmallRng, prefix: u32) -> Option<Id> {
        use rand::RngCore;
        let step = rng.next_u32() % Self::RND_MAX_STEP;
        let new_id = id.unprefix().checked_add(step)?;
        Some(Id(new_id | prefix))
    }

    /// Remove all prefix bits from the ID.
    pub const fn unprefix(self) -> u32 {
        self.0 & !(Self::NODE_PREFIX | Self::EDGE_PREFIX)
    }

    pub const fn get(&self) -> u32 {
        self.0
    }
}

#[derive(Debug)]
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
        tuples: SmallVec<[(Pat, Value); 1]>,
        wildcard: Option<Value>,
    },

    /// List of values of some type. Can be used for filtering or
    /// values validation (e.g. to find invalid/unexpected values).
    List { values: SmallVec<[Value; 1]> },

    /// Check the boolean predicate and execute either branch.
    IfElse { condition: Predicate },

    /// Check Result and execute either branch passing the corresponding value.
    OkOrErr,

    /// Match the input values and execute appropriate branch.
    /// Match should be exhaustive (or with wildcard).
    Match {
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
    Todo {
        /// Message to display when crashing, can be empty (then ignored).
        msg: CompactString,
    },

    /// Just some comment, for visuals. Can have many inputs and outputs,
    /// which just directly pass the value(s) forward. Must have the same number of ins and outs.
    Comment(CompactString),
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
            SplitBy { .. } => &[Some(PT::Array(&[PT::Record]))],
            Input { .. } => &[None],
            Drop => &[],
            Output { .. } => &[],
            StrOp(_) => &[Some(PT::Str)],
            Compare { .. } => &[Some(PT::Bool)],
            Ordering => &[Some(PT::Ordering)],
            Regex(_) => &[None],
            FindRecord(_) => &[Some(PT::Result(&(PT::Array(&[PT::Record]), PT::Unit)))],
            Map { .. } => &[None],
            List { .. } => &[None],
            IfElse { .. } => &[None, None],
            OkOrErr => &[None, None],
            Match { .. } => return None,
            Func(_) => return None,
            ParseDateTime { date, time, .. } => match (date, time) {
                (true, false) => &[Some(PT::Result(&(PT::Date, PT::Unit)))],
                (false, true) => &[Some(PT::Result(&(PT::Time, PT::Unit)))],
                (true, true) => &[Some(PT::Result(&(PT::DateTime, PT::Unit)))],
                (false, false) => &[Some(PT::Result(&(PT::Unit, PT::Unit)))],
            },
            ParseMonetary { .. } => &[Some(PT::Result(&(PT::Moneraty, PT::Unit)))],
            ParseInt { .. } => &[Some(PT::Result(&(PT::Int, PT::Unit)))],
            Constant(_) => &[None],
            Crash { .. } => &[],
            OkOrCrash { .. } => &[None],
            Unreachable { .. } => &[],
            Todo { .. } => &[],
            Comment(_) => &[None],
        };
        Some(some)
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
            Map { .. } => &[None],
            List { .. } => &[None],
            IfElse { .. } => &[None],
            OkOrErr => &[None],
            Match { .. } => &[None],
            Func(_) => return None,
            ParseDateTime { .. } => &[Some(PT::Str)],
            ParseMonetary { .. } => &[Some(PT::Str)],
            ParseInt { .. } => &[Some(PT::Str)],
            Constant(_) => &[],
            Crash { .. } => &[None],
            OkOrCrash { .. } => &[None],
            Unreachable { .. } => &[None],
            Todo { .. } => &[None],
            Comment(_) => &[None],
        };
        Some(some)
    }

    /// Whether the node is a root node.
    pub const fn is_root(&self) -> bool {
        matches!(self, NodeStub::File)
    }
}

#[derive(Debug)]
pub struct Pat {}

#[derive(Debug)]
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
    Result(Box<(Value, Value)>),
    Option(Box<Value>),
}

#[derive(Debug)]
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
    Result(Box<(PrimitiveType, PrimitiveType)>),
    Option(Box<PrimitiveType>),
}

#[derive(Debug)]
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
    Array(&'static [PrimitiveTypeConst]),
    Predicate(&'static PredicateConst),
    Result(&'static (PrimitiveTypeConst, PrimitiveTypeConst)),
    Option(&'static PrimitiveTypeConst),
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
#[derive(Debug)]
pub struct Predicate {
    pub inputs: SmallVec<[PrimitiveType; 1]>,
    pub outputs: SmallVec<[PrimitiveType; 1]>,
}

#[derive(Debug)]
pub struct PredicateConst {
    pub inputs: &'static [PrimitiveType],
    pub outputs: &'static [PrimitiveType],
}

/// Operation to apply to a `String` input.
#[derive(Debug)]
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
