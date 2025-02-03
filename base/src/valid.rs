use std::marker::PhantomData;

use smallvec::SmallVec;

use crate::canvas::{self, Canvas};

struct AssignedType {
    /// Assigned type of the edge. Can be empty if not known, or
    /// can have many types if the edge is ambiguous.
    ty: SmallVec<[canvas::PrimitiveType; 1]>,
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

    pub fn resolve_types(&mut self) -> &Typed<'canvas> {
        if let Some(types) = &self.types {
            return types;
        }

        todo!()
    }
}

impl<T> Canvas<T> {
    fn adjacent_child_nodes(
        &self,
        node: canvas::NodeIdx,
    ) -> impl Iterator<Item = canvas::NodeIdx> + '_ {
        let start = canvas::EdgeInner::binary_search_from(self, node) as usize;
        self.edges[start..]
            .iter()
            .take_while(move |edge| edge.from.0 == node)
            .map(|edge| edge.to.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use canvas::Pin;

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

        canvas
            .add_edge(Pin::only_node_id(a), Pin::only_node_id(b))
            .unwrap();
        canvas
            .add_edge(Pin::only_node_id(b), Pin::only_node_id(c))
            .unwrap();
        canvas
            .add_edge(Pin::only_node_id(c), Pin::only_node_id(a))
            .unwrap();

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

        canvas
            .add_edge(Pin::only_node_id(a), Pin::only_node_id(b))
            .unwrap();
        canvas
            .add_edge(Pin::only_node_id(b), Pin::only_node_id(c))
            .unwrap();
        canvas
            .add_edge(Pin::only_node_id(c), Pin::only_node_id(d))
            .unwrap();
        canvas
            .add_edge(Pin::only_node_id(d), Pin::only_node_id(b))
            .unwrap();

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

        canvas
            .add_edge(Pin::only_node_id(a), Pin::only_node_id(b))
            .unwrap();
        canvas
            .add_edge(Pin::only_node_id(b), Pin::only_node_id(c))
            .unwrap();
        canvas
            .add_edge(Pin::only_node_id(c), Pin::only_node_id(d))
            .unwrap();
        canvas
            .add_edge(Pin::only_node_id(d), Pin::only_node_id(b))
            .unwrap();
        canvas
            .add_edge(Pin::only_node_id(c), Pin::only_node_id(e))
            .unwrap();

        let mut validator = Validator::new(&canvas);
        let cycles = validator.detect_cycles();
        assert_eq!(cycles.cycles.len(), 1);

        let cycle = &cycles.cycles[0];
        println!("{cycle:#?}");
        assert!(Cycles::same(cycle, &[3, 2, 1]));
    }
}
