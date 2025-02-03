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
            let mut stack = Stack::new();
            let mut cycles = SmallVec::<[_; 8]>::new();

            for root in self.canvas.root_nodes.iter().copied() {
                stack.visit(self.canvas, root, &mut cycles);
            }

            Cycles {
                _canvas: PhantomData,
                cycles: cycles.into_vec(),
            }
        });

        #[derive(Debug, Clone)]
        struct Stack {
            stack: SmallVec<[canvas::NodeIdx; 64]>,
        }

        struct CycleDetectedError;

        impl Stack {
            fn new() -> Self {
                Self {
                    stack: SmallVec::new(),
                }
            }

            fn insert(&mut self, node: canvas::NodeIdx) -> Result<(), CycleDetectedError> {
                if self.contains(node) {
                    Err(CycleDetectedError)
                } else {
                    self.stack.push(node);
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
                cycles: &mut SmallVec<[Vec<canvas::EdgeIdx>; 8]>,
            ) {
                // NOTE: reimplementation for performance reasons can be considered.

                if self.insert(node).is_err() {
                    // We found a cycle.
                    cycles.push(self.stack.to_vec());
                } else {
                    for adj in canvas.adjacent_child_nodes(node) {
                        self.visit(canvas, adj, cycles);
                    }
                }
                self.stack.pop();
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
