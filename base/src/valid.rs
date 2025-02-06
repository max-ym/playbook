use std::{collections::VecDeque, marker::PhantomData};

use log::{debug, info, trace};
use smallvec::SmallVec;

use crate::canvas::{self, Canvas};

#[derive(Debug, Clone)]
struct AssignedType {
    /// Assigned type of the edge. Can be empty if not known, or
    /// can have many types if the edge is ambiguous.
    ty: SmallVec<[canvas::PrimitiveType; 1]>,

    /// Mark the type as erroneous, e.g. when the statically known types of
    /// input and output pins do not match.
    is_err: bool,
}

impl AssignedType {
    pub fn is_empty(&self) -> bool {
        self.ty.is_empty()
    }
}

/// Assigned types for the canvas nodes.
pub struct Typed<'canvas> {
    _canvas: PhantomData<&'canvas Canvas<()>>,

    /// This array corresponds by position to the edges in the canvas.
    /// Since we take reference to the canvas, we know that as long as this
    /// object is alive, the canvas cannot be modified.
    edges: Vec<AssignedType>,
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

        struct Resolver<'validator, 'canvas, T> {
            validator: &'validator Validator<'canvas, T>,
            edges: Vec<AssignedType>,
            resolve_next: VecDeque<canvas::NodeIdx>,

            // Buffer used to represent pin types of nodes, for
            // node-level resolver from [canvas] module.
            buf: Vec<Option<canvas::PrimitiveType>>,
        }

        impl<'validator, 'canvas: 'validator, T> Resolver<'validator, 'canvas, T> {
            pub fn new(validator: &'validator Validator<'canvas, T>) -> Self {
                Self {
                    validator,
                    edges: vec![
                        AssignedType {
                            ty: SmallVec::new(),
                            is_err: false,
                        };
                        validator.canvas.edges.len()
                    ],

                    // Preallocate big enough queue for possible nodes.
                    resolve_next: VecDeque::with_capacity(validator.canvas.nodes.len().min(512)),
                    buf: Vec::with_capacity(64),
                }
            }

            pub fn finish(self) -> Typed<'canvas> {
                Typed {
                    _canvas: PhantomData,
                    edges: self.edges,
                }
            }

            fn canvas(&self) -> &'canvas Canvas<T> {
                self.validator.canvas
            }

            pub fn resolve_all(&mut self) {
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
                            assert!(
                                result.is_progres(),
                                "Ok result here should mean progress was made"
                            );
                            trace!("progress was made for node {node_idx}");
                            self.save_buf_for(node_idx);
                            continue;
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

            fn load_buf_for(&mut self, node_idx: canvas::NodeIdx) {
                trace!("loading types resolver buffer for node {node_idx}");
                let stub = &self.canvas().nodes[node_idx as usize].stub;

                self.buf.clear();
                self.buf
                    .resize_with(stub.total_pin_count(), Default::default);

                // Find all edges and load their currently-known types.
                let node_edges = &self.validator.node_edges[node_idx as usize];
                for edge_idx in node_edges.arr.iter().copied() {
                    let edge = &self.canvas().edges[edge_idx as usize];
                    let ty = &self.edges[edge_idx as usize];

                    let pin_idx = if edge.from.0 == node_idx {
                        edge.from.1
                    } else {
                        edge.to.1
                    };

                    self.buf[pin_idx as usize] = if ty.is_err {
                        None
                    } else {
                        Some(ty.ty[0].clone())
                    };
                }
            }

            fn save_buf_for(&mut self, node: canvas::NodeIdx) {
                trace!("saving types from types resolver buffer for node {node}");

                let node_edges = &self.validator.node_edges[node as usize];
                for edge_idx in node_edges.arr.iter().copied() {
                    let edge = &self.canvas().edges[edge_idx as usize];
                    let assigned_ty = &mut self.edges[edge_idx as usize];

                    let pin_idx = if edge.from.0 == node {
                        edge.from.1
                    } else {
                        edge.to.1
                    };

                    assigned_ty.ty.clear();
                    if let Some(ty) = &self.buf[pin_idx as usize] {
                        assigned_ty.ty.push(ty.clone());
                    }
                }
            }

            /// Mark this node as erroneous.
            fn mark_node_err(&mut self, node: canvas::NodeIdx) {
                trace!("marking all {node} node's edges as erroneous");
                let node_edges = &self.validator.node_edges[node as usize];
                for edge_idx in node_edges.arr.iter().copied() {
                    let assigned_ty = &mut self.edges[edge_idx as usize];
                    assigned_ty.is_err = true;
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
}
