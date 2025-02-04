use std::{collections::VecDeque, marker::PhantomData};

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

pub struct Validator<'canvas, NodeMeta> {
    canvas: &'canvas Canvas<NodeMeta>,

    /// Result of cycle detection if it was performed.
    cycles: Option<Cycles<'canvas>>,

    /// Assigned types for the canvas nodes, if they were assigned.
    types: Option<Typed<'canvas>>,
}

impl<'canvas, NodeMeta> Validator<'canvas, NodeMeta> {
    pub fn new(canvas: &'canvas Canvas<NodeMeta>) -> Self {
        Self {
            canvas,
            cycles: None,
            types: None,
        }
    }

    /// Detect cycles in the canvas.
    pub fn detect_cycles(&mut self) -> &Cycles<'canvas> {
        return self.cycles.get_or_insert_with(|| {
            // We search as such:
            // Take any root node, and do a DFS search.
            // If we find a cycle, we mark it and continue with other unvisited nodes.
            let mut visitor = Visitor::new(&self.canvas);
            let mut cycles = SmallVec::<[_; 8]>::new();

            for root in self.canvas.root_nodes.iter().copied() {
                visitor.visit(self.canvas, root, &mut cycles);
            }

            // Also account for nodes that have no root nodes because they are cyclic between
            // each other with no head or tail.
            let mut last = 0;
            while last != visitor.unvisited.len() {
                if visitor.unvisited[last] {
                    visitor.visit(self.canvas, last as canvas::NodeIdx, &mut cycles);
                }
                last += 1;
            }

            Cycles {
                _canvas: PhantomData,
                cycles: cycles.into_vec(),
            }
        });

        #[derive(Debug, Clone)]
        struct Visitor {
            stack: SmallVec<[canvas::NodeIdx; 64]>,

            /// Map of visited nodes.
            unvisited: Vec<bool>,
        }

        struct CycleDetectedError;

        impl Visitor {
            fn new<T>(canvas: &Canvas<T>) -> Self {
                Self {
                    stack: SmallVec::new(),
                    unvisited: vec![true; canvas.nodes.len()],
                }
            }

            fn insert(&mut self, node: canvas::NodeIdx) -> Result<(), CycleDetectedError> {
                if self.contains(node) {
                    Err(CycleDetectedError)
                } else {
                    self.stack.push(node);
                    self.unvisited[node as usize] = false;
                    Ok(())
                }
            }

            fn contains(&self, node: canvas::NodeIdx) -> bool {
                self.stack.contains(&node)
            }

            fn visit<T>(
                &mut self,
                canvas: &Canvas<T>,
                node: canvas::NodeIdx,
                cycles: &mut SmallVec<[Vec<canvas::NodeIdx>; 8]>,
            ) {
                // NOTE: reimplementation for performance reasons can be considered.

                if self.insert(node).is_err() {
                    // We found a cycle.
                    cycles.push(self.collect_cycle(node));
                } else {
                    for adj in canvas.adjacent_child_nodes(node) {
                        self.visit(canvas, adj, cycles);
                    }
                }
                self.stack.pop();
            }

            // Collect a cycle from current stack, backtracing.
            // Panics if the cycle head is not found in the stack.
            fn collect_cycle(&self, head: canvas::NodeIdx) -> Vec<canvas::NodeIdx> {
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
        if self.detect_cycles().is_some() {
            return None;
        }

        return Some(self.types.get_or_insert_with(|| {
            let mut r = Resolver::new(self.canvas);
            r.resolve_all();
            r.finish()
        }));

        struct Resolver<'canvas, T> {
            canvas: &'canvas Canvas<T>,
            edges: Vec<AssignedType>,
            resolve_next: VecDeque<canvas::NodeIdx>,

            // Buffer used to represent pin types of nodes, for
            // node-level resolver from [canvas] module.
            buf: Vec<Option<canvas::PrimitiveType>>,
        }

        impl<'canvas, T> Resolver<'canvas, T> {
            pub fn new(canvas: &'canvas Canvas<T>) -> Self {
                Self {
                    canvas,
                    edges: vec![
                        AssignedType {
                            ty: SmallVec::new(),
                            is_err: false,
                        };
                        canvas.edges.len()
                    ],

                    // Preallocate big enough queue for possible edges.
                    resolve_next: VecDeque::with_capacity(canvas.edges.len().min(512)),
                    buf: Vec::with_capacity(64),
                }
            }

            pub fn finish(self) -> Typed<'canvas> {
                Typed {
                    _canvas: PhantomData,
                    edges: self.edges,
                }
            }

            pub fn resolve_all(&mut self) {
                // We start from the root nodes and then propagate the types.
                self.resolve_next
                    .extend(self.canvas.root_nodes.iter().copied());

                while let Some(next) = self.resolve_next.pop_back() {
                    self.resolve(next);
                }
            }

            fn resolve(&mut self, node_idx: canvas::NodeIdx) {
                use canvas::ResolvePinTypes;
                use std::convert::Infallible as Never;

                let stub = &self.canvas.nodes[node_idx as usize].stub;

                loop {
                    use canvas::PinResolutionError::*;

                    self.load_buf_for(stub);
                    let result = ResolvePinTypes::resolve(stub, &mut self.buf, true);
                    let _: Never = match result {
                        Ok(result) => {
                            assert!(
                                result.is_progres(),
                                "Ok result here should mean progress was made"
                            );
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
                            // Save any existing info for possible debug.
                            self.save_buf_for(node_idx);
                            self.mark_node_err(node_idx);

                            // We would not be able to resolve this node, so we stop here.
                            break;
                        }
                        Err(RemainingUnknownPins) => {
                            // No progress was made, so we end with this node.
                            break;
                        }
                    };
                }
            }

            fn load_buf_for(&mut self, node: &canvas::NodeStub) {
                self.buf.clear();
                self.buf
                    .resize_with(node.total_pin_count(), Default::default);

                // Find all edges and load their currently-known types.
                todo!()
            }

            fn save_buf_for(&mut self, node: canvas::NodeIdx) {
                todo!()
            }

            /// Mark this node as erroneous.
            fn mark_node_err(&mut self, node: canvas::NodeIdx) {
                todo!()
            }
        }
    }
}

impl<T> Canvas<T> {
    fn adjacent_child_nodes(
        &self,
        node: canvas::NodeIdx,
    ) -> impl Iterator<Item = canvas::NodeIdx> + '_ {
        self.adjacent_child_edges(node)
            .map(move |edge| self.edges[edge as usize].to.0)
    }

    fn adjacent_child_edges(
        &self,
        node: canvas::NodeIdx,
    ) -> impl Iterator<Item = canvas::EdgeIdx> + '_ {
        let start = canvas::EdgeInner::binary_search_from(self, node) as usize;
        self.edges[start..]
            .iter()
            .enumerate()
            .take_while(move |&(_, edge)| edge.from.0 == node)
            .map(move |(index, _)| (start + index) as canvas::EdgeIdx)
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
        assert_eq!(cycles.cycles.len(), 1);

        let cycle = &cycles.cycles[0];
        println!("{cycle:#?}");
        assert!(Cycles::same(cycle, &[3, 2, 1]));
    }

    #[test]
    fn detect_cycle_middle() {
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
