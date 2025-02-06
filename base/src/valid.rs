use std::{collections::VecDeque, marker::PhantomData};

use log::{debug, info, trace};
use smallvec::SmallVec;

use crate::canvas::{self, Canvas};

#[derive(Debug, Clone)]
struct AssignedType {
    /// Assigned type of the edge. Can be empty if not known, or
    /// can have many types if the edge is ambiguous.
    ty: Option<canvas::PrimitiveType>,

    /// Mark the type as erroneous, e.g. when the statically known types of
    /// input and output pins do not match.
    is_err: bool,
}

impl AssignedType {
    pub fn is_empty(&self) -> bool {
        self.ty.is_none()
    }
}

/// Assigned types for the canvas nodes.
#[derive(Debug)]
pub struct Typed<'canvas> {
    _canvas: PhantomData<&'canvas Canvas<()>>,

    /// Unique types defined in the canvas.
    tys: Vec<canvas::PrimitiveType>,

    /// Assigned types for the node pins.
    /// Inner array is an array of indices into the [Self::tys] array.
    /// For unresolved pins, the index is [usize::MAX].
    nodes: Vec<Vec<usize>>,
}

/// Cycles in the canvas.
#[derive(Debug)]
pub struct Cycles<'canvas> {
    _canvas: PhantomData<&'canvas Canvas<()>>,

    /// Cycles in the canvas.
    cycles: Vec<Vec<canvas::EdgeIdx>>,
}

impl Cycles<'_> {
    /// Whether both cycles contain all the same edges in the same order.
    /// Even if the initial node is different, the cycle is considered the same.
    pub fn same(this: &[canvas::EdgeIdx], other: &[canvas::EdgeIdx]) -> bool {
        if this.len() != other.len() {
            return false;
        }

        // Iterate until we find the first edge that is the same.
        if let Some(first) = other.iter().position(|&x| x == this[0]) {
            let mut other_iter = other.iter().cycle().skip(first);
            for &edge in this.iter() {
                if edge != *other_iter.next().unwrap() {
                    return false;
                }
            }
        }

        true
    }

    pub fn is_some(&self) -> bool {
        !self.cycles.is_empty()
    }
}

/// Collected input and output edges for single node, to speed up lookup
/// for the resolver.
#[derive(Debug, Clone, Default)]
struct NodeEdges {
    /// Array containing all input and output edge indices.
    /// First go input edges, then output edges.
    /// Each input edge is sorted by source node index,
    /// and each output edge is sorted by destination node index.
    /// Sorted slices allow for binary search.
    arr: SmallVec<[canvas::EdgeIdx; 16]>,

    /// Index where the output edges start.
    split_at: canvas::EdgeIdx,
}

impl NodeEdges {
    fn inputs(&self) -> &[canvas::EdgeIdx] {
        &self.arr[..self.split_at as usize]
    }

    fn inputs_mut(&mut self) -> &mut [canvas::EdgeIdx] {
        &mut self.arr[..self.split_at as usize]
    }

    fn outputs(&self) -> &[canvas::EdgeIdx] {
        &self.arr[self.split_at as usize..]
    }

    fn outputs_mut(&mut self) -> &mut [canvas::EdgeIdx] {
        &mut self.arr[self.split_at as usize..]
    }
}

/// Validator for the whole canvas. Validates all the nodes and edges
/// in the canvas.
pub struct Validator<'canvas, NodeMeta> {
    /// Reference to the canvas being validated.
    canvas: &'canvas Canvas<NodeMeta>,

    /// Lookup table for node edges. Helps to quickly find adjacent nodes.
    /// Is empty if collection was not performed.
    /// Each entry corresponds to the nodes array in the canvas.
    node_edges: Vec<NodeEdges>,

    /// Result of cycle detection, if it was performed.
    cycles: Option<Cycles<'canvas>>,

    /// Assigned types for the canvas nodes, if they were assigned.
    types: Option<Typed<'canvas>>,
}

impl<'canvas, NodeMeta> Validator<'canvas, NodeMeta> {
    pub fn new(canvas: &'canvas Canvas<NodeMeta>) -> Self {
        Self {
            canvas,
            node_edges: Vec::new(),
            cycles: None,
            types: None,
        }
    }

    /// Fill in the lookup table for node edges.
    /// This is used to speed up the resolver.
    ///
    /// This does nothing if the table is already filled.
    pub fn collect_node_edges_table(&mut self) {
        if self.node_edges.len() > 0 {
            assert_eq!(
                self.node_edges.len(),
                self.canvas.nodes.len(),
                "node edges table should be filled for all nodes",
            );
            return;
        }

        self.node_edges = vec![NodeEdges::default(); self.canvas.nodes.len()];

        info!("fill in raw node-edges lookup table entries");
        for (edge_idx, edge) in self.canvas.edges.iter().enumerate() {
            let from = edge.from.0 as usize;
            let to = edge.to.0 as usize;

            self.node_edges[from].arr.push(edge_idx as canvas::EdgeIdx);
            self.node_edges[to].arr.push(edge_idx as canvas::EdgeIdx);
        }

        // Sort the edges for each node.
        // We need each output and input sub-arrays to be sorted by the
        // destination or source node index, respectively.
        debug!("sort node-edges lookup table entries");
        for (idx, node_edges) in self.node_edges.iter_mut().enumerate() {
            trace!("sort node-edges for node {idx}");
            node_edges.arr.sort_by_key(|&edge| {
                let edge = &self.canvas.edges[edge as usize];
                if edge.to.0 == idx as canvas::NodeIdx {
                    0
                } else {
                    1
                }
            });

            // Find the split point for input/output edges.
            node_edges.split_at = node_edges
                .arr
                .iter()
                .position(|&edge| {
                    let edge = &self.canvas.edges[edge as usize];
                    edge.to.0 != idx as canvas::NodeIdx
                })
                .map(|v| v as canvas::EdgeIdx)
                .unwrap_or_else(|| node_edges.arr.len() as canvas::EdgeIdx);
            trace!("split at {}", node_edges.split_at);

            trace!("sort the input edges by source node index");
            node_edges.inputs_mut().sort_by_key(|&edge| {
                let edge = &self.canvas.edges[edge as usize];
                edge.from.0
            });

            trace!("sort the output edges by destination node index");
            node_edges.outputs_mut().sort_by_key(|&edge| {
                let edge = &self.canvas.edges[edge as usize];
                edge.to.0
            });
        }
        debug!("node-edges lookup table filled");
    }

    fn is_node_edges_ready(&self) -> bool {
        self.node_edges.len() == self.canvas.nodes.len()
    }

    fn adjacent_child_nodes(
        &self,
        node: canvas::NodeIdx,
    ) -> impl Iterator<Item = canvas::NodeIdx> + '_ {
        assert!(self.is_node_edges_ready());
        return Iter {
            prev: None,
            iter: self.node_edges[node as usize].outputs().iter().copied(),
            validator: self,
        };

        // This iterator to help us deduplicate the output.
        // Since the edges are sorted by destination node index,
        // we can skip duplicates by just comparing subsequent edge indices.
        struct Iter<'canvas, NodeMeta, I: Iterator<Item = canvas::EdgeIdx>> {
            prev: Option<canvas::NodeIdx>,
            iter: I,
            validator: &'canvas Validator<'canvas, NodeMeta>,
        }

        impl<T, I: Iterator<Item = canvas::EdgeIdx>> Iterator for Iter<'_, T, I> {
            type Item = canvas::NodeIdx;

            fn next(&mut self) -> Option<Self::Item> {
                while let Some(edge) = self.iter.next() {
                    let edge = self.validator.canvas.edges[edge as usize];
                    let node = edge.to.0;
                    if Some(node) != self.prev {
                        self.prev = Some(node);
                        return Some(node);
                    }
                }
                None
            }
        }
    }

    /// Detect cycles in the canvas.
    pub fn detect_cycles(&mut self) -> &Cycles<'canvas> {
        return {
            if self.cycles.is_none() {
                self.collect_node_edges_table();
                info!("detect cycles in the canvas");

                // We search as such:
                // Take any root node, and do a DFS search.
                // If we find a cycle, we mark it and continue with other unvisited nodes.
                let mut visitor = Visitor::new(&self.canvas);
                let mut cycles = SmallVec::<[_; 8]>::new();

                debug!("detect cycles in the canvas for flows from root nodes");
                for root in self.canvas.root_nodes.iter().copied() {
                    visitor.visit(self, root, &mut cycles);
                }

                // Also account for nodes that have no root nodes because they are cyclic between
                // each other with no head or tail.
                debug!("detect cycles in the canvas for remaining nodes");
                let mut last = 0;
                while last != visitor.visited.len() {
                    if !visitor.visited[last] {
                        visitor.visit(self, last as canvas::NodeIdx, &mut cycles);
                    }
                    last += 1;
                }

                self.cycles = Some(Cycles {
                    _canvas: PhantomData,
                    cycles: cycles.into_vec(),
                });
            }
            self.cycles
                .as_ref()
                .expect("either initialized just above, or already was initialized on call")
        };

        #[derive(Debug, Clone)]
        struct Visitor {
            stack: SmallVec<[canvas::NodeIdx; 64]>,

            /// Map of visited nodes.
            visited: Vec<bool>,
        }

        struct CycleDetectedError;

        impl Visitor {
            fn new<T>(canvas: &Canvas<T>) -> Self {
                Self {
                    stack: SmallVec::new(),
                    visited: vec![false; canvas.nodes.len()],
                }
            }

            fn insert(&mut self, node: canvas::NodeIdx) -> Result<(), CycleDetectedError> {
                if self.contains(node) {
                    Err(CycleDetectedError)
                } else {
                    self.stack.push(node);
                    self.visited[node as usize] = true;
                    Ok(())
                }
            }

            fn contains(&self, node: canvas::NodeIdx) -> bool {
                self.stack.contains(&node)
            }

            fn visit<T>(
                &mut self,
                validator: &Validator<T>,
                node: canvas::NodeIdx,
                cycles: &mut SmallVec<[Vec<canvas::NodeIdx>; 8]>,
            ) {
                // NOTE: reimplementation for performance reasons can be considered.

                if self.insert(node).is_err() {
                    debug!("found cycle in the canvas");
                    cycles.push(self.collect_cycle(node));
                } else {
                    for adj in validator.adjacent_child_nodes(node) {
                        self.visit(validator, adj, cycles);
                    }
                }
                self.stack.pop();
            }

            // Collect a cycle from current stack, backtracing.
            // Panics if the cycle head is not found in the stack.
            fn collect_cycle(&self, head: canvas::NodeIdx) -> Vec<canvas::NodeIdx> {
                trace!("collect cycle for head node {head}");
                let mut rewind_stack = SmallVec::<[_; 64]>::new();
                let mut iter = self.stack.iter().copied().rev();
                while let Some(next) = iter.next() {
                    rewind_stack.push(next);
                    if next == head {
                        break;
                    }
                }
                assert_eq!(rewind_stack.last(), Some(&head));
                rewind_stack.into_vec()
            }
        }
    }

    /// Resolve types for the canvas nodes.
    /// This will return None if there are cycles in the canvas.
    pub fn resolve_types(&mut self) -> Option<&Typed<'canvas>> {
        return {
            if self.types.is_none() {
                if self.detect_cycles().is_some() {
                    return None;
                }

                info!("resolve types for the canvas");

                let mut r = Resolver::new(self);
                r.resolve_all();
                self.types = Some(r.finish());
            }
            self.types.as_ref()
        };

        struct PinMap {
            node_idx: canvas::NodeIdx,

            // Pin number of the node.
            pin: canvas::PinOrder,

            // Index into 'pins' array.
            map_idx: usize,
        }

        impl std::cmp::PartialEq for PinMap {
            fn eq(&self, other: &Self) -> bool {
                self.node_idx == other.node_idx && self.pin == other.pin
            }
        }

        impl std::cmp::Eq for PinMap {}

        impl std::cmp::PartialOrd for PinMap {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl std::cmp::Ord for PinMap {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.node_idx
                    .cmp(&other.node_idx)
                    .then(self.pin.cmp(&other.pin))
            }
        }

        struct Resolver<'validator, 'canvas, T> {
            validator: &'validator Validator<'canvas, T>,

            // Representation of each individual pin in the canvas,
            // except for those with edges being merged into one pin,
            // as part of normalization, for type assignment.
            pins: Vec<AssignedType>,

            // Sorted array of node pins indexes, binary searchable.
            node_to_pins: Vec<PinMap>,

            // Buffer for the next nodes to resolve.
            resolve_next: VecDeque<canvas::NodeIdx>,

            // Buffer used to represent pin types of nodes, for
            // node-level resolver from [canvas] module.
            buf: Vec<Option<canvas::PrimitiveType>>,
        }

        impl<'validator, 'canvas: 'validator, T> Resolver<'validator, 'canvas, T> {
            pub fn new(validator: &'validator Validator<'canvas, T>) -> Self {
                let total_pins_in_canvas: usize = validator
                    .canvas
                    .nodes
                    .iter()
                    .map(|node| node.stub.total_pin_count())
                    .sum();
                Self {
                    validator,
                    pins: Vec::new(),
                    node_to_pins: Vec::with_capacity(total_pins_in_canvas),

                    // Preallocate big enough queue for possible nodes.
                    resolve_next: VecDeque::with_capacity(validator.canvas.nodes.len().min(512)),
                    buf: Vec::with_capacity(64),
                }
            }

            fn init_pins(&mut self) {
                if !self.pins.is_empty() {
                    assert!(!self.node_to_pins.is_empty());
                    return;
                }

                for (node_idx, node) in self.validator.canvas.nodes.iter().enumerate() {
                    let stub = &node.stub;
                    for pin in 0..stub.total_pin_count() {
                        self.node_to_pins.push(PinMap {
                            node_idx: node_idx as canvas::NodeIdx,
                            pin: pin as canvas::PinOrder,
                            map_idx: usize::MAX, // tmp invalid value
                        });
                    }
                }

                debug_assert!(self.node_to_pins.is_sorted());

                self.pins = vec![
                    AssignedType {
                        ty: None,
                        is_err: false,
                    };
                    self.node_to_pins.len() - self.canvas().edges.len()
                ];

                trace!("init pins for all nodes, which appear in an edge");
                let mut cnt = 0;
                for edge in self.canvas().edges.iter() {
                    let pin_idx = if edge.from.0 == edge.to.0 {
                        edge.from.1
                    } else {
                        edge.to.1
                    };
                    let result = self.node_to_pins.binary_search(&PinMap {
                        node_idx: edge.from.0,
                        pin: pin_idx,
                        map_idx: usize::MAX,
                    });
                    if let Ok(i) = result {
                        self.node_to_pins[i].map_idx = cnt;
                        cnt += 1;
                    } else {
                        // Already initialized.
                        debug_assert!(self
                            .node_to_pins
                            .binary_search_by(|probe| {
                                probe
                                    .node_idx
                                    .cmp(&edge.from.0)
                                    .then_with(|| probe.pin.cmp(&pin_idx))
                            })
                            .is_ok());
                    }
                }

                trace!("init pins that do not appear in an edge");
                for pin in self.node_to_pins.iter_mut() {
                    if pin.map_idx == usize::MAX {
                        pin.map_idx = cnt;
                        cnt += 1;
                    }
                }

                debug_assert!(self
                    .node_to_pins
                    .iter()
                    .all(|pin| pin.map_idx != usize::MAX));
            }

            fn pin_ty_mut(
                &mut self,
                node_idx: canvas::NodeIdx,
                pin: canvas::PinOrder,
            ) -> &mut AssignedType {
                debug_assert!(!self.pins.is_empty());

                let idx = self
                    .node_to_pins
                    .binary_search_by(|probe| {
                        probe
                            .node_idx
                            .cmp(&node_idx)
                            .then_with(|| probe.pin.cmp(&pin))
                    })
                    .expect("pin should be defined in the map by this point");
                let map = &self.node_to_pins[idx];
                let mapped_idx = map.map_idx;

                trace!("resolved pin {pin} for node {node_idx} as map idx {mapped_idx}");
                &mut self.pins[mapped_idx]
            }

            pub fn finish(self) -> Typed<'canvas> {
                todo!()
            }

            fn canvas(&self) -> &'canvas Canvas<T> {
                self.validator.canvas
            }

            pub fn resolve_all(&mut self) {
                self.init_pins();

                // We start from the root nodes and then propagate the types.
                self.resolve_next
                    .extend(self.canvas().root_nodes.iter().copied());

                while let Some(next) = self.resolve_next.pop_back() {
                    self.resolve(next);
                }
            }

            fn resolve(&mut self, node_idx: canvas::NodeIdx) {
                use canvas::ResolvePinTypes;
                use std::convert::Infallible as Never;

                debug!("resolve types for node {node_idx}");

                let stub = &self.canvas().nodes[node_idx as usize].stub;

                loop {
                    use canvas::PinResolutionError::*;

                    self.load_buf_for(node_idx);
                    let result = ResolvePinTypes::resolve(stub, &mut self.buf, true);
                    let _: Never = match result {
                        Ok(result) => {
                            let is_progress = result.is_progress();
                            self.save_buf_for(node_idx);
                            if is_progress {
                                trace!("progress was made for node {node_idx}");
                                continue;
                            } else {
                                trace!("fully resolved node {node_idx}");
                                break;
                            }
                        }
                        Err(PinNumberMismatch) => {
                            unreachable!("pin number mismatch, invalid preallocation?");
                            // Pins should have been preallocated per node requirements, and
                            // if we reach this point, it means that particular code
                            // likely has a bug.
                        }
                        Err(UnionConflict) => {
                            debug!("union conflict for node {node_idx}");
                            // Save any existing info for possible debug.
                            self.save_buf_for(node_idx);
                            self.mark_node_err(node_idx);

                            // We would not be able to resolve this node, so we stop here.
                            break;
                        }
                        Err(RemainingUnknownPins) => {
                            trace!("remaining unknown pins for node {node_idx}");
                            // No progress was made, so we end with this node.
                            break;
                        }
                    };
                }
            }

            fn node_pins(&self, node_idx: canvas::NodeIdx) -> &[PinMap] {
                trace!("get pins for node {node_idx}");
                debug_assert!(!self.node_to_pins.is_empty());
                debug_assert!(self.node_to_pins.is_sorted());

                let start = self
                    .node_to_pins
                    .binary_search_by(|probe| {
                        probe
                            .node_idx
                            .cmp(&node_idx)
                            .then_with(|| probe.pin.cmp(&canvas::PinOrder::MIN))
                    })
                    .expect("node pins should be defined in the map by this point");
                let end = match self.node_to_pins.binary_search_by(|probe| {
                    probe
                        .node_idx
                        .cmp(&(node_idx))
                        .then_with(|| probe.pin.cmp(&canvas::PinOrder::MAX))
                }) {
                    Ok(end) => end + 1,
                    Err(end) => end,
                };
                trace!("pins range for {node_idx} is {start}..{end}");
                &self.node_to_pins[start..end]
            }

            fn load_buf_for(&mut self, node_idx: canvas::NodeIdx) {
                trace!("loading types resolver buffer for node {node_idx}");
                let stub = &self.canvas().nodes[node_idx as usize].stub;

                self.buf.clear();
                self.buf
                    .resize_with(stub.total_pin_count(), Default::default);

                let mut buf = std::mem::take(&mut self.buf);
                for pin_map in self.node_pins(node_idx) {
                    let pin_idx = pin_map.map_idx;
                    trace!("mapped pin to index {pin_idx}");
                    let ty = &self.pins[pin_idx].ty;
                    buf[pin_map.pin as usize] = ty.to_owned();
                    trace!("loaded type {ty:?} for pin {} to buffer", pin_map.pin);
                }

                std::mem::swap(&mut self.buf, &mut buf);
                trace!("loaded types resolver buffer for node {node_idx} as {:?}", self.buf);
            }

            fn save_buf_for(&mut self, node: canvas::NodeIdx) {
                trace!("saving types from types resolver buffer for node {node}");

                let mut buf = std::mem::take(&mut self.buf);
                for (pin_idx, ty) in buf.drain(..).enumerate() {
                    trace!("save type {ty:?} for pin {pin_idx}");
                    self.pin_ty_mut(node, pin_idx as canvas::PinOrder).ty = ty;
                }

                std::mem::swap(&mut self.buf, &mut buf);
            }

            /// Mark this node as erroneous.
            fn mark_node_err(&mut self, node: canvas::NodeIdx) {
                // TODO this can be done simpler since we have 'pins' array now.

                trace!("marking all {node} node's edges as erroneous");
                let node_edges = &self.validator.node_edges[node as usize];
                for edge_idx in node_edges.arr.iter().copied() {
                    let edge = &self.canvas().edges[edge_idx as usize];
                    let pin_idx = if edge.from.0 == node {
                        edge.from.1
                    } else {
                        edge.to.1
                    };
                    self.pin_ty_mut(node, pin_idx).is_err = true;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use canvas::Pin;

    macro_rules! outpin {
        ($node:expr) => {
            canvas::OutputPin(Pin::only_node_id($node))
        };
    }
    macro_rules! inpin {
        ($node:expr) => {
            canvas::InputPin(Pin::only_node_id($node))
        };
    }

    // Test cycle detection.
    #[test]
    fn detect_cycle_dangling() {
        crate::tests::init();

        let mut canvas = crate::canvas::Canvas::new();
        let node = canvas::NodeStub::Todo {
            msg: Default::default(),
            inputs: 0,
        };

        let a = canvas.add_node(node.clone(), ());
        let b = canvas.add_node(node.clone(), ());
        let c = canvas.add_node(node.clone(), ());

        canvas.add_edge(outpin!(a), inpin!(b)).unwrap();
        canvas.add_edge(outpin!(b), inpin!(c)).unwrap();
        canvas.add_edge(outpin!(c), inpin!(a)).unwrap();

        let mut validator = Validator::new(&canvas);
        let cycles = validator.detect_cycles();
        assert_eq!(cycles.cycles.len(), 1);

        let cycle = &cycles.cycles[0];
        println!("{cycle:#?}");
        assert!(Cycles::same(cycle, &[2, 1, 0]));
    }

    #[test]
    fn detect_cycle() {
        crate::tests::init();

        let mut canvas = crate::canvas::Canvas::new();
        let node = canvas::NodeStub::Todo {
            msg: Default::default(),
            inputs: 0,
        };

        let a = canvas.add_node(node.clone(), ());
        let b = canvas.add_node(node.clone(), ());
        let c = canvas.add_node(node.clone(), ());
        let d = canvas.add_node(node.clone(), ());

        canvas.add_edge(outpin!(a), inpin!(b)).unwrap();
        canvas.add_edge(outpin!(b), inpin!(c)).unwrap();
        canvas.add_edge(outpin!(c), inpin!(d)).unwrap();
        canvas.add_edge(outpin!(d), inpin!(b)).unwrap();

        let mut validator = Validator::new(&canvas);
        let cycles = validator.detect_cycles();
        println!("{cycles:#?}");
        assert_eq!(cycles.cycles.len(), 1);

        let cycle = &cycles.cycles[0];
        println!("{cycle:#?}");
        assert!(Cycles::same(cycle, &[3, 2, 1]));
    }

    #[test]
    fn detect_cycle_middle() {
        crate::tests::init();

        let mut canvas = crate::canvas::Canvas::new();
        let node = canvas::NodeStub::Todo {
            msg: Default::default(),
            inputs: 0,
        };

        let a = canvas.add_node(node.clone(), ());
        let b = canvas.add_node(node.clone(), ());
        let c = canvas.add_node(node.clone(), ());
        let d = canvas.add_node(node.clone(), ());
        let e = canvas.add_node(node.clone(), ());

        canvas.add_edge(outpin!(a), inpin!(b)).unwrap();
        canvas.add_edge(outpin!(b), inpin!(c)).unwrap();
        canvas.add_edge(outpin!(c), inpin!(d)).unwrap();
        canvas.add_edge(outpin!(d), inpin!(b)).unwrap();
        canvas.add_edge(outpin!(c), inpin!(e)).unwrap();

        let mut validator = Validator::new(&canvas);
        let cycles = validator.detect_cycles();
        assert_eq!(cycles.cycles.len(), 1);

        let cycle = &cycles.cycles[0];
        println!("{cycle:#?}");
        assert!(Cycles::same(cycle, &[3, 2, 1]));
    }

    #[test]
    fn resolve0() {
        crate::tests::init();

        let mut canvas = crate::canvas::Canvas::new();
        let input_stub = canvas::NodeStub::Input {
            valid_names: Default::default(),
        };
        let parse_int_stub = canvas::NodeStub::ParseInt { signed: false };
        let ordering_stub = canvas::NodeStub::Ordering;

        let input0 = canvas.add_node(input_stub.clone(), ());
        let input1 = canvas.add_node(input_stub, ());
        let parse0 = canvas.add_node(parse_int_stub.clone(), ());
        let parse1 = canvas.add_node(parse_int_stub, ());
        let order = canvas.add_node(ordering_stub, ());

        canvas.add_edge(outpin!(input0), inpin!(parse0)).unwrap();
        canvas.add_edge(outpin!(input1), inpin!(parse1)).unwrap();
        canvas.add_edge(outpin!(parse0), inpin!(order)).unwrap();
        canvas.add_edge(outpin!(parse1), inpin!(order)).unwrap();

        let mut validator = Validator::new(&canvas);
        let types = validator.resolve_types().unwrap();
        println!("{:#?}", types);
    }
}
