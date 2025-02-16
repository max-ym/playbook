use super::*;

use hashbrown::HashSet;
use log::{error, info, trace};
use smallvec::SmallVec;

#[derive(Debug, Clone)]
pub(crate) struct AssignedType {
    /// Assigned type of the edge. Can be [Ok] but [None] if not known.
    /// If the type is [Err], it means that the type is erroneous and cannot be resolved.
    ty: Result<HintedPrimitiveType, ()>,
}

impl Default for AssignedType {
    fn default() -> Self {
        Self::new()
    }
}

impl AssignedType {
    pub const fn new() -> Self {
        Self {
            ty: Ok(HintedPrimitiveType::Hint),
        }
    }

    pub const fn with(ty: HintedPrimitiveType) -> Self {
        Self { ty: Ok(ty) }
    }

    pub fn is_some(&self) -> bool {
        self.ty
            .as_ref()
            .map(|v| matches!(v, HintedPrimitiveType::Hint))
            .unwrap_or(false)
    }

    pub fn is_none(&self) -> bool {
        !self.is_some()
    }

    /// Whether the type is fully resolved: is known, not erroneous and have no
    /// hint marks in the inner types, if any.
    pub fn is_resolved(&self) -> bool {
        self.ty.as_ref().map_or(false, |ty| ty.is_resolved())
    }

    pub fn is_err(&self) -> bool {
        self.ty.is_err()
    }

    pub fn is_ok(&self) -> bool {
        self.ty.is_ok()
    }

    pub fn mark_err(&mut self) {
        self.ty = Err(());
    }

    pub fn set_ty(&mut self, ty: HintedPrimitiveType) {
        if self.is_ok() {
            self.ty = Ok(ty);
        }
    }

    /// Clarify the type using the given type. Returns false if the operation was
    /// successful or not.
    /// It will return true even if prevision did not impore (e.g. types were the same).
    /// If this type is already in error, it will not be changed and false will be returned.
    /// Uncompatible types will cause the current type to be marked as erroneous.
    pub fn union_type_with(&mut self, ty: HintedPrimitiveType) -> TypePrecision {
        use TypePrecision::*;

        match &mut self.ty {
            Ok(prev) => match prev.clarify(&ty) {
                (_, Same) => Same,
                (_, Uncertain) => Uncertain,
                (ty, Precise) => {
                    self.set_ty(ty);
                    Precise
                }
                (_, Incompatible) => {
                    self.mark_err();
                    Incompatible
                }
            },
            Err(_) => Incompatible,
        }
    }

    /// Make both variables to be the same type. They are compared on compatibility if both
    /// are present, and are clarified if any of them is less precise to be the most precise
    /// common type. True is returned on change, false if no change was made.
    ///
    /// # Erroneous
    /// The erroneous types are not changed and false is returned.
    /// This keeps healthy types from being changed to erroneous types, which may
    /// promote better type resolution as best-effort scenario.
    fn union(a: &mut AssignedType, b: &mut AssignedType) -> Result<bool, UnionConflict> {
        if a.is_err() || b.is_err() {
            return Ok(false);
        }

        let ty = Self::any(a, b)?;
        let prec = a.union_type_with(ty);
        match prec {
            TypePrecision::Precise => {
                b.ty = a.ty.clone();
                Ok(true)
            }
            TypePrecision::Same | TypePrecision::Uncertain => Ok(false),
            TypePrecision::Incompatible => Err(UnionConflict),
        }
    }

    /// The same as [AssignedType::union] but with the ability to poison the second type
    /// if the union is incompatible.
    /// If "safe" type is already in error, the "poisoned" type also gets poisoned.
    fn poisoning_union(
        safe: &AssignedType,
        poisoned: &mut AssignedType,
    ) -> Result<bool, UnionConflict> {
        if safe.is_err() {
            let is_poisoned = poisoned.is_err();
            return if !is_poisoned {
                poisoned.mark_err();
                Ok(true)
            } else {
                Ok(false)
            };
        } else {
            let ty = Self::any(safe, poisoned)?;
            let precision = poisoned.union_type_with(ty);
            match precision {
                TypePrecision::Precise => {
                    poisoned.ty = safe.ty.clone();
                    Ok(true)
                }
                TypePrecision::Same | TypePrecision::Uncertain => Ok(false),
                TypePrecision::Incompatible => Err(UnionConflict),
            }
        }
    }

    /// Set all pins to this exact given value. Return true if any change was made.
    ///
    /// # Erroneous
    /// The erroneous types are not changed and false is returned on no change at both
    /// types.
    fn set_to(val: AssignedType, a: &mut AssignedType, b: &mut AssignedType) -> bool {
        let mut changed = false;
        if a.is_ok() && a.ty != val.ty {
            a.ty = val.ty.clone();
            changed = true;
        }
        if b.is_ok() && b.ty != val.ty {
            b.ty = val.ty;
            changed = true;
        }
        changed
    }

    /// If `b` is known, it should be the same or be more precise than `a`.
    /// The write is performed if that improves the precision in `b`.
    /// True is returned if change was made.
    ///
    /// # Erroneous
    /// The erroneous types are not changed and `Ok(false)` is returned.
    fn match_or_write(a: HintedPrimitiveType, b: &mut AssignedType) -> Result<bool, UnionConflict> {
        if b.is_err() {
            return Ok(false);
        }

        let prec = b.union_type_with(a);
        match prec {
            TypePrecision::Precise => Ok(true),
            TypePrecision::Same | TypePrecision::Uncertain => Ok(false),
            TypePrecision::Incompatible => Err(UnionConflict),
        }
    }

    /// Return the new, most precise type calculated from both present types.
    /// If any is unknown, the other is returned. If both are unknown,
    /// [HintedPrimitiveType::Hint] is returned.
    ///
    /// # Erroneous
    /// Erroneous types are ignored and treated as if they were not present.
    fn any(a: &AssignedType, b: &AssignedType) -> Result<HintedPrimitiveType, UnionConflict> {
        use TypePrecision::*;

        match (&a.ty, &b.ty) {
            (Ok(a), Ok(b)) => {
                let (val, prec) = a.clarify(b);
                match prec {
                    Precise | Same | Uncertain => Ok(val),
                    Incompatible => Err(UnionConflict),
                }
            }
            (Ok(val), Err(_)) | (Err(_), Ok(val)) => Ok(val.clone()),
            (Err(_), Err(_)) => Ok(HintedPrimitiveType::Hint),
        }
    }

    pub fn opt_ty(&self) -> Option<&HintedPrimitiveType> {
        self.ty.as_ref().ok()
    }

    pub fn to_ty_or_hint(&self) -> HintedPrimitiveType {
        self.opt_ty().cloned().unwrap_or(HintedPrimitiveType::Hint)
    }
}

trait IsProgress {
    fn is_progress(&self) -> bool;
}

impl IsProgress for Result<bool, UnionConflict> {
    fn is_progress(&self) -> bool {
        match self {
            Ok(progress) => *progress,
            Err(_) => true,
        }
    }
}

/// Error that can happen when querying the pin type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PinQueryError {
    /// Node does not (anymore) exist in the canvas.
    NodeNotFound(Id),

    /// Pin does not exist in the node.
    PinNotFound(Pin),
}

/// Cycles in the canvas.
#[derive(Debug, Clone)]
pub struct Cycle(Vec<Edge>);

impl Cycle {
    /// Whether both cycles contain all the same edges in the same order.
    /// Even if the initial node is different, the cycle is considered the same.
    pub fn same(&self, other: &Cycle) -> bool {
        todo!()
    }
}

/// Validator for the canvas.
#[derive(Debug)]
pub(crate) struct Validator {
    /// Buffer for type resolution algorithms, so to not reallocate the same
    /// memory over and over again for each subsequent run, for each node.
    buf: Vec<AssignedType>,

    /// Buffer for tracking nodes in queue for resolution.
    resolv_stack: ResolutionStack,

    /// Set of nodes with modifications that require revalidation.
    nodes_modified: HashSet<Id>,

    /// Nodes known to validator.
    /// Sorted by ID.
    nodes: Vec<NodeData>,

    /// Pin groups - pins that are known to share the type (because they are connected).
    pin_groups: Vec<PinGroup>,
}

/// Node-specific data in the validator.
#[derive(Debug)]
pub(crate) struct NodeData {
    /// ID of the node.
    id: Id,

    ty: SmallVec<[PinGroupIdx; 2]>,
}

type PinGroupIdx = usize;

/// Pins that are grouped together because the edges between them make them the same type,
/// hence they have the same type assigned.
#[derive(Debug, Default)]
pub(crate) struct PinGroup(AssignedType);

impl Deref for PinGroup {
    type Target = AssignedType;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for PinGroup {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> Canvas<T> {
    /// Get pin type for given pin. Returns [Ok] of [None] if the pin has no resolved type.
    /// Returns [Ok] with [Some] resolved type. Returns [Err] if the node/pin does not exist.
    pub fn calc_pin_type(&mut self, pin: Pin) -> Result<Option<PrimitiveType>, PinQueryError> {
        Validator::resolve_pin_type(self, pin)?;
        self.pin_type(pin)
    }

    /// Get pin type for given pin. Returns [Ok] of [None] if the pin has no resolved type, or
    /// if the type is only partially resolved. Returns [Ok] with [Some] resolved type.
    /// Returns [Err] if the node/pin does not exist.
    pub fn pin_type(&self, pin: Pin) -> Result<Option<PrimitiveType>, PinQueryError> {
        if let Some(ty) = Validator::peek_pin_type(self, pin)? {
            if ty.is_resolved() {
                Ok(Some(ty.into()))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }
}

impl Validator {
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(128),
            resolv_stack: ResolutionStack::preallocated(),
            nodes_modified: HashSet::with_capacity(256),
            nodes: Vec::with_capacity(256),
            pin_groups: Vec::with_capacity(256),
        }
    }

    fn node(&self, id: Id) -> Option<&NodeData> {
        self.node_idx(id).map(|idx| &self.nodes[idx as usize])
    }

    fn node_idx(&self, id: Id) -> Option<NodeIdx> {
        self.nodes
            .binary_search_by(|n| n.id.cmp(&id))
            .ok()
            .map(|v| v as NodeIdx)
    }

    /// Peek into the pin type. If it is not known, return [None].
    pub(crate) fn peek_pin_type<T>(
        canvas: &Canvas<T>,
        pin: Pin,
    ) -> Result<Option<&HintedPrimitiveType>, PinQueryError> {
        let is_modified = canvas.valid.nodes_modified.contains(&pin.node_id);
        if is_modified {
            return Ok(None);
        }

        let node = canvas
            .valid
            .node(pin.node_id)
            .ok_or(PinQueryError::NodeNotFound(pin.node_id))?;
        let pin_group_idx = node.ty[pin.order as usize];
        let pin_group = &canvas.valid.pin_groups[pin_group_idx];
        Ok(pin_group.opt_ty())
    }

    fn set_pin_type<T>(canvas: &mut Canvas<T>, pin: Pin, ty: AssignedType) {
        let node = canvas
            .valid
            .node(pin.node_id)
            .expect("existence proved by caller");
        let pin_group_idx = node.ty[pin.order as usize];
        let pin_group = &mut canvas.valid.pin_groups[pin_group_idx];
        *pin_group = PinGroup(ty);
    }

    fn clear_node_pins(&mut self, node_idx: usize) {
        // Take out the pin groups array for modification.
        let mut pin_groups = std::mem::take(&mut self.pin_groups);

        for &pin_group_idx in self.nodes[node_idx].ty.iter() {
            pin_groups[pin_group_idx] = PinGroup::default();
        }

        // Return the modified pin groups.
        self.pin_groups = pin_groups;
    }

    pub fn resolve_pin_type<T>(
        canvas: &mut Canvas<T>,
        pin: Pin,
    ) -> Result<&NodeData, PinQueryError> {
        let mut nodes_modified = std::mem::take(&mut canvas.valid.nodes_modified);
        for modified in nodes_modified.drain() {
            let idx = canvas
                .node_id_to_idx(modified)
                .expect("correct maintenance of the nodes_modified set");
            canvas.valid.resolv_stack.unstuck(idx);
        }
        canvas.valid.nodes_modified = nodes_modified;

        // Next node to resolve.
        canvas.valid.resolv_stack.push(
            canvas
                .node_id_to_idx(pin.node_id)
                .ok_or(PinQueryError::NodeNotFound(pin.node_id))?,
        );

        while let Some(node_idx) = canvas.valid.resolv_stack.peek() {
            let node_idx = node_idx as usize;
            if canvas
                .valid
                .nodes_modified
                .contains(&canvas.nodes[node_idx].id)
            {
                trace!("node at {node_idx} was modified, clearing old resolution");
                canvas.valid.clear_node_pins(node_idx);
                canvas
                    .valid
                    .nodes_modified
                    .remove(&canvas.nodes[node_idx].id);
            }

            Self::resolve_node(canvas);
        }

        Ok(canvas
            .valid
            .node(pin.node_id)
            .expect("should exist due to ran validation"))
    }

    /// Resolve types for the pins of a node comming first in the queue.
    /// The function can again enqueue the node if it is not fully resolved,
    /// but is believed to proceed once other nodes
    /// in the queue are processed.
    fn resolve_node<T>(canvas: &mut Canvas<T>) {
        use std::convert::Infallible as Never;

        let node_idx =
            canvas.valid.resolv_stack.next().expect(
                "this function should only be called when there are enqueued nodes to resolve",
            ) as usize;
        let node_id = canvas.nodes[node_idx].id;
        let buf_release_tok = Validator::load_buf_for(canvas, node_id);

        trace!("entering type resolution loop for node {node_id}");
        loop {
            use PinResolutionError::*;

            let stub = &canvas.nodes[node_idx].stub;
            let result = ResolvePinTypes::resolve(stub, &mut canvas.valid.buf);
            let _: Never = match result {
                Ok(result) => {
                    let is_progress = result.is_progress();
                    let is_resolved = result.is_resolved();

                    if is_resolved {
                        trace!("fully resolved node {node_idx}");
                        break;
                    } else if is_progress {
                        trace!("progress was made for node {node_idx}");
                        canvas.valid.resolv_stack.unstuck(node_idx as NodeIdx);
                        continue;
                    } else {
                        trace!("no progress was made for node {node_idx}");

                        // Put this node into wait queue, run its neighbors.
                        // It may give more information for this node to resolve.
                        canvas
                            .valid
                            .resolv_stack
                            .push_await_maybe_stuck(node_idx as NodeIdx);
                        Validator::neighbors_scan_for_revalidation(node_idx, canvas);
                        return;
                    }
                }
                Err(PinNumberMismatch) => {
                    unreachable!("pin number mismatch, invalid preallocation?");
                    // Pins should have been preallocated per node requirements, and
                    // if we reach this point, it means that particular code
                    // likely has a bug.
                }
            };
        }

        Validator::save_buf_for(canvas, buf_release_tok);
        Validator::neighbors_scan_for_revalidation(node_idx, canvas);
    }

    /// Check whether neighbors of the node connected by the edges require revalidation, and if
    /// so, add them to the `resolve_next` queue.
    ///
    /// True is returned if at least one neighbor requires revalidation and was added to the queue.
    fn neighbors_scan_for_revalidation<T>(node_idx: usize, canvas: &mut Canvas<T>) -> bool {
        let node_id = canvas.nodes[node_idx].id;
        let mut require_others = false;
        let mut resolve_stack = std::mem::take(&mut canvas.valid.resolv_stack);
        for edge in canvas
            .node_edge_io_iter(node_id)
            .expect("existence proved by caller")
        {
            let other_node_id = edge.oppose(node_id).node_id;
            if canvas.valid.nodes_modified.contains(&other_node_id) {
                require_others = true;
                let idx = canvas
                    .node_id_to_idx(other_node_id)
                    .expect("correct edge maintenance on add/remove operations");
                resolve_stack.push(idx);
            }
        }
        canvas.valid.resolv_stack = resolve_stack;
        require_others
    }

    /// Load type resolution buffer for the node.
    #[must_use = "buffer should be explicitly saved or discarded"]
    fn load_buf_for<T>(canvas: &mut Canvas<T>, node_id: Id) -> BufferReleaseToken {
        use std::mem;

        let pin_cnt = canvas
            .node(node_id)
            .expect("existence proved by caller")
            .stub
            .total_pin_count() as usize;

        trace!("preload default (unresolved) types for node {node_id} pins");
        canvas.valid.buf.clear();
        canvas.valid.buf.resize_with(pin_cnt, Default::default);

        // We take the buffer out of the validator, so we can modify it
        // at the same time as we borrow canvas (which contains validator) for edge iteration.
        let mut buf = mem::take(&mut canvas.valid.buf);

        trace!("fill in known types for node {node_id} pins from the existing edges");
        for edge in canvas
            .node_edge_io_iter(node_id)
            .expect("existence proved by caller")
        {
            let other = edge.oppose(node_id);
            let ty = Validator::peek_pin_type(canvas, other)
                .expect("correct edge maintenance guarantee existance of required nodes");
            buf[other.order as usize].ty = Ok(ty.cloned().unwrap_or(HintedPrimitiveType::Hint));
        }
        canvas.valid.buf = buf;

        BufferReleaseToken(node_id)
    }

    /// Save changes from type resolution buffer for the node.
    /// It must be the same node that was loaded last time.
    fn save_buf_for<T>(canvas: &mut Canvas<T>, release: BufferReleaseToken) {
        use std::mem;

        // We take the buffer out of the validator, so we can drain it, to prevent copying types
        // and then return back. Similarly, we take the pin groups out of the validator for edits.
        let mut buf = mem::take(&mut canvas.valid.buf);
        let mut pin_groups = mem::take(&mut canvas.valid.pin_groups);

        let node = canvas
            .valid
            .node(release.0)
            .expect("existence proved by caller");

        for (pin_grp_idx, assigned) in node.ty.iter().copied().zip(buf.drain(..)) {
            pin_groups[pin_grp_idx] = PinGroup(assigned);
        }

        // Return what we took.
        canvas.valid.buf = buf;
        canvas.valid.pin_groups = pin_groups;
    }
}

/// Stack to hold information about resolution nodes that are planned to be resolved next.
/// This is used during resolution to track order for current node resolution request.
#[derive(Debug)]
struct ResolutionStack {
    /// Next planned node to resolve.
    next: Vec<NodeIdx>,

    /// If [Self::next] is empty, this stack is used to resolve nodes.
    /// Nodes are put here when they themselves were triggering the resolution
    /// of other nodes by putting them into [Self::next]. This way, their resolution was
    /// postponed until all other nodes were resolved, to gather more information required
    /// for complete resolution.
    awaiting: Vec<NodeIdx>,

    /// Nodes that are stuck in the resolution loop, so we won't run them again.
    maybe_stuck: HashSet<NodeIdx>,
}

impl Iterator for ResolutionStack {
    type Item = NodeIdx;

    fn next(&mut self) -> Option<Self::Item> {
        self.next.pop().or_else(|| self.awaiting.pop())
    }
}

impl ExactSizeIterator for ResolutionStack {
    fn len(&self) -> usize {
        self.next.len() + self.awaiting.len()
    }
}

impl Default for ResolutionStack {
    fn default() -> Self {
        Self::new()
    }
}

impl ResolutionStack {
    pub fn new() -> Self {
        Self {
            next: Vec::new(),
            awaiting: Vec::new(),
            maybe_stuck: HashSet::new(),
        }
    }

    /// Create a preallocated stack with hardcoded capacity.
    pub fn preallocated() -> Self {
        Self {
            next: Vec::with_capacity(256),
            awaiting: Vec::with_capacity(32),
            maybe_stuck: HashSet::with_capacity(256),
        }
    }

    /// Push node to the stack to be resolved next, unless it is marked as stuck.
    pub fn push(&mut self, node: NodeIdx) {
        if self.maybe_stuck.contains(&node) {
            return;
        }
        self.next.push(node);
    }

    /// Push node to the stack to be resolved next, unless it is marked as stuck.
    pub fn push_await(&mut self, node: NodeIdx) {
        if self.maybe_stuck.contains(&node) {
            return;
        }
        self.awaiting.push(node);
    }

    /// Push node to the stack to be resolved next, unless it was already marked as stuck.
    /// This will mark the node as stuck, so it won't be resolved
    /// again unless [unstuck](Self::unstuck).
    pub fn push_await_maybe_stuck(&mut self, node: NodeIdx) {
        self.push_await(node);
        self.maybe_stuck.insert(node);
    }

    /// Mark node as not stuck, so it can be resolved again.
    pub fn unstuck(&mut self, node: NodeIdx) {
        self.maybe_stuck.remove(&node);
    }

    /// Peek into the next node to resolve.
    pub fn peek(&self) -> Option<NodeIdx> {
        self.next.last().or_else(|| self.awaiting.last()).copied()
    }
}

/// Release token to remind to save or discard the buffer.
#[derive(Debug)]
struct BufferReleaseToken(Id);

impl BufferReleaseToken {
    #[allow(dead_code)]
    pub fn discard(self) {}
}

impl Default for Validator {
    fn default() -> Self {
        Self::new()
    }
}

impl Edge {
    /// Get the opposite side of the edge that connects the given node.
    fn oppose(self, this_node_id: Id) -> Pin {
        if self.from.node_id == this_node_id {
            *self.to
        } else {
            debug_assert_eq!(self.to.node_id, this_node_id);
            *self.from
        }
    }
}

/// Result of resolution of pin types of some node.
pub struct ResolvePinTypes<'a> {
    pins: &'a mut Vec<AssignedType>,

    // TODO: instead indicate if further relaunch of the same node is needed right now as believed
    // to advance resolution further. To avoid running the same node over one more time
    // when it is determined to provide no more new information at this iteration in global
    // resolution loop.
    is_progress: bool,
}

impl<'pins> ResolvePinTypes<'pins> {
    pub(crate) fn pin_io_slices_mut<'a>(
        pins: &'a mut [AssignedType],
        node: &NodeStub,
    ) -> (&'a mut [AssignedType], &'a mut [AssignedType]) {
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
        pins: &'pins mut Vec<AssignedType>,
    ) -> Result<Self, PinResolutionError> {
        if pins.len() as PinOrder != node.total_pin_count() {
            error!(
                "Pin number mismatch: ({}/{})",
                pins.len(),
                node.total_pin_count()
            );
            return Err(PinResolutionError::PinNumberMismatch);
        }
        if node.total_pin_count() == 0 {
            // Fuse for nodes without pins.
            // This prevents issues with those rules that expect there's at least
            // something to resolve.
            // Such nodes can happen when, for example, Match node has no variants not
            // wildcard, so effectively it does not receive or output any data.
            return Ok(Self {
                pins,
                is_progress: false,
            });
        }

        let mut is_progress = ResolvePinTypes::prefill_with_static(node, pins)?;

        let result = {
            match node.static_io_relation() {
                IoRelation::Same(pairs) => {
                    for &(i, o) in pairs {
                        let (i, o) = (i as usize, o as usize);
                        let (i, o) = {
                            // To satisfy the borrow checker, we split slice to guarantee
                            // non-overlapping mutable references.
                            let (a, b) = pins.split_at_mut(i + 1);
                            (&mut a[i], &mut b[o - i - 1])
                        };

                        is_progress |= AssignedType::union(i, o).is_progress();
                    }

                    Self { pins, is_progress }
                }
                IoRelation::FullSymmetry => {
                    let (ins, outs) = Self::pin_io_slices_mut(pins, node);
                    if ins.len() != outs.len() {
                        // If the number of inputs and outputs is different, we cannot
                        // resolve the types.
                        return Err(PinResolutionError::PinNumberMismatch);
                    }
                    for (i, o) in ins.iter_mut().zip(outs.iter_mut()) {
                        is_progress |= AssignedType::union(i, o).is_progress();
                    }

                    Self { pins, is_progress }
                }
                IoRelation::Dynamic => {
                    use NodeStub::*;
                    match node {
                        Regex { .. } => {
                            // All should be strings.
                            for pin in pins.iter_mut() {
                                is_progress |=
                                    AssignedType::match_or_write(HintedPrimitiveType::Str, pin)
                                        .is_progress();
                            }

                            Self { pins, is_progress }
                        }
                        Map { tuples, wildcard } => {
                            let input = tuples
                                .first()
                                .map(|tup| tup.0.value())
                                .or(wildcard.as_ref())
                                .map(Value::type_of);
                            let output = tuples
                                .first()
                                .map(|tup| &tup.1)
                                .or(wildcard.as_ref())
                                .map(Value::type_of);

                            let (ins, outs) = Self::pin_io_slices_mut(pins, node);
                            debug_assert_eq!(ins.len(), 1);
                            debug_assert_eq!(outs.len(), 1);

                            if let Some(input) = input {
                                is_progress |=
                                    AssignedType::match_or_write(input.into(), &mut ins[0])
                                        .is_progress();
                            }
                            if let Some(output) = output {
                                is_progress |=
                                    AssignedType::match_or_write(output.into(), &mut outs[0])
                                        .is_progress();
                            }

                            // Input types should be already provided externally.
                            Self { pins, is_progress }
                        }
                        IfElse { .. } => {
                            let (ins, outs) = Self::pin_io_slices_mut(pins, node);
                            is_progress |= AssignedType::match_or_write(
                                HintedPrimitiveType::Bool,
                                &mut ins[0],
                            )
                            .ok()
                            .unwrap_or(true);
                            for (i, o) in ins.iter_mut().skip(1).zip(outs.iter_mut()) {
                                is_progress |= AssignedType::union(i, o).is_progress();
                            }

                            Self { pins, is_progress }
                        }
                        Validate { .. } => {
                            // All outputs are wrapped as Result, but otherwise, are the same as
                            // the inputs.
                            let (ins, outs) = Self::pin_io_slices_mut(pins, node);
                            for (i, o) in ins.iter_mut().zip(outs.iter_mut()) {
                                if let Some(ty) = i.opt_ty() {
                                    trace!("Validate: input known, propagate");
                                    let out = HintedPrimitiveType::Result(Box::new((
                                        ty.clone(),
                                        ty.clone(),
                                    )));
                                    is_progress |=
                                        AssignedType::match_or_write(out, o).is_progress();
                                } else if let Some(HintedPrimitiveType::Result(boxed)) = o.opt_ty()
                                {
                                    // We expect output pin to be Result<T, T> where T is the
                                    // type of the input pin that we should propagate.
                                    trace!("Validate: output known, propagate");
                                    let (ok, err) = &**boxed;
                                    let (clarified, prec) = ok.clarify(err);
                                    is_progress |= match prec {
                                        TypePrecision::Precise
                                        | TypePrecision::Same
                                        | TypePrecision::Uncertain => {
                                            AssignedType::match_or_write(clarified, i).is_progress()
                                        }
                                        TypePrecision::Incompatible => {
                                            trace!("Validate: expected Result<T, T> output, but found incompatible types");
                                            let is_err = o.is_err();
                                            if !is_err {
                                                o.mark_err();
                                                true
                                            } else {
                                                false
                                            }
                                        }
                                    };
                                } else {
                                    trace!("Validate: unknown I/O pin pair types");
                                }
                            }

                            Self { pins, is_progress }
                        }
                        SelectFirst { .. } => {
                            // Inputs can be Option or non-Option type. If Option input exists,
                            // it's inner type should be the same as non-Option one. If
                            // non Option is not provided, all types should be the same.
                            // There is just one output of that selected type. It will be
                            // Optional if and only if all inputs are optional.

                            let (ins, outs) = Self::pin_io_slices_mut(pins, node);

                            // Validate and mark invalid pins, which will help for
                            // better resolution.
                            // 1. First input and output pin should have the same type.
                            // 2. All other input pins should be Options.
                            // 3. All pins should have the same inner value. If first input and
                            //    output are not an Option, they should be this inner type.

                            // #1
                            trace!("SelectFirst 1: check first input pin and output pin have the same type");
                            if let (Some(i), Some(o)) = (ins.first_mut(), outs.last_mut()) {
                                let p = AssignedType::union(i, o).is_progress();
                                if p {
                                    trace!("Resolution progress of 1st in and out pins: {i:?}")
                                }
                                is_progress |= p;
                            }

                            // #2
                            trace!(
                                "SelectFirst 2: all except the first input pins should be Options"
                            );
                            let mut opt = AssignedType::with(HintedPrimitiveType::Option(
                                Box::new(HintedPrimitiveType::Unit),
                            ));
                            let mut local_progress = false;
                            for _ in 0..=1 {
                                for i in ins.iter_mut().skip(1) {
                                    let p = AssignedType::union(i, &mut opt).is_progress();
                                    if p {
                                        trace!("Resolution progress of Option pin: {i:?}");
                                    }
                                    local_progress |= p;
                                }
                                if !local_progress {
                                    // Second iteration won't propagate any new information.
                                    trace!("Second iteration not needed");
                                    break;
                                } else {
                                    // 'opt' now has the best possible information about
                                    // the inner type of the Option. We will re-run the
                                    // loop to populate all pins with this information (except
                                    // for erroneous ones).
                                    trace!("Second iteration will be ran");
                                }
                            }
                            is_progress |= local_progress;

                            let is_first_opt = ins.first().map(|i| {
                                matches!(i.opt_ty(), Some(HintedPrimitiveType::Option(_)))
                            });
                            let is_out_opt = outs.last().map(|o| {
                                matches!(o.opt_ty(), Some(HintedPrimitiveType::Option(_)))
                            });
                            trace!(
                                "First input is Option: {is_first_opt:?}, output is Option: {is_out_opt:?}"
                            );

                            // #3
                            trace!("SelectFirst 3: all pins should have the same inner type");
                            let mut inner = 'inner: {
                                // At this point, "opt" variable has high chance of being
                                // already resolved.
                                if opt.is_resolved() {
                                    trace!("Option type is resolved, unwrap its inner type");
                                    // Unwrap from option and return.
                                    if let HintedPrimitiveType::Option(inner) = opt.to_ty_or_hint()
                                    {
                                        trace!("Resolved inner type: {inner:#?}");
                                        break 'inner AssignedType::with(*inner);
                                    } else {
                                        unreachable!(
                                            "resolved type should be Option in this position"
                                        );
                                    }
                                }

                                // If the first input is not an Option, we can use it as the inner type.
                                trace!("Option type is not resolved, try to use the first input pin's type");
                                if let Some(false) = is_first_opt {
                                    if let Some(ty) = ins.first().and_then(|i| i.opt_ty()) {
                                        trace!("Used first input pin as inner type: {ty:#?}");
                                        break 'inner AssignedType::with(ty.to_owned());
                                    } else {
                                        // Maybe pin is in error?
                                        trace!(
                                            "Not used first input pin due to invalid associated state"
                                        );
                                    }
                                } else {
                                    trace!("Not used first input pin due to unuseful types");
                                }

                                trace!(
                                    "Option type is not resolved, try to use the output pin's type"
                                );
                                if let Some(false) = is_out_opt {
                                    if let Some(ty) = outs.last().and_then(|o| o.opt_ty()) {
                                        trace!("Used output pin as inner type: {ty:#?}");
                                        break 'inner AssignedType::with(ty.to_owned());
                                    } else {
                                        // Maybe pin is in error?
                                        trace!(
                                            "Not used output pin as inner type due to invalid associated state"
                                        );
                                    }
                                } else {
                                    trace!(
                                        "Not used output pin as inner type due to unuseful types"
                                    );
                                }

                                trace!("Inner type is not resolved");
                                AssignedType::default()
                            };

                            // Try to give more precise information about the inner type to
                            // our Option-holding variable.
                            if let Some(inner) = inner.opt_ty() {
                                trace!(
                                    "There's some information about inner type for propagation into Option variant: {inner:#?}"
                                );
                                is_progress |= AssignedType::union(
                                    &mut AssignedType::with(HintedPrimitiveType::Option(Box::new(
                                        inner.to_owned(),
                                    ))),
                                    &mut opt,
                                )
                                .is_progress();
                            }

                            // Propagate the inner type (without a wrapper when applicable) to all pins.
                            trace!("Propagate inner type to all pins");

                            // Repeat option union as our option should now be the best guess.
                            trace!("Repeat Option union for all pins");
                            for i in ins.iter_mut().skip(1) {
                                is_progress |= AssignedType::union(i, &mut opt).is_progress();
                            }
                            // Set the correct first pin, which may or may not be option.
                            match is_first_opt {
                                Some(true) => {
                                    trace!("First input pin is Option");
                                    is_progress |=
                                        AssignedType::union(ins.first_mut().unwrap(), &mut opt)
                                            .is_progress();
                                }
                                Some(false) => {
                                    trace!("First input pin is not Option");
                                    is_progress |=
                                        AssignedType::union(ins.first_mut().unwrap(), &mut inner)
                                            .is_progress();
                                }
                                None => {
                                    // Do nothing, as we don't know the first pin type for sure,
                                    // it could be either Option or not.
                                    trace!("First input pin type is not evident");
                                }
                            }
                            match is_out_opt.or(is_first_opt) {
                                Some(true) => {
                                    trace!("Output pin is Option");
                                    is_progress |=
                                        AssignedType::union(&mut outs[0], &mut opt).is_progress();
                                }
                                Some(false) => {
                                    trace!("Output pin is not Option");
                                    is_progress |=
                                        AssignedType::union(&mut outs[0], &mut inner).is_progress();
                                }
                                None => {
                                    // Do nothing, as we don't know the output pin type for sure,
                                    // it could be either Option or not.
                                    trace!("Output pin type is not evident");
                                }
                            }

                            trace!("SelectFirst summary: progress {is_progress}, {pins:#?}");
                            Self { pins, is_progress }
                        }
                        ExpectOne { .. } => {
                            // All inputs should be the same type, and the output is the result:
                            // Result<type, Array<type>>.

                            let (ins, outs) = Self::pin_io_slices_mut(pins, node);
                            assert_eq!(outs.len(), 1);

                            trace!("ExpectOne: get a known type in any of the input pins");
                            let mut detect = AssignedType::new();
                            for i in ins.iter_mut() {
                                is_progress |= AssignedType::union(&mut detect, i).is_progress();
                            }
                            trace!("Detected type for propagation: {detect:#?}");

                            trace!("Propagete detected type to all input pins");
                            for i in ins.iter_mut() {
                                is_progress |= AssignedType::union(&mut detect, i).is_progress();
                            }

                            // Output should be Result<type, Array<type>>.
                            trace!("Propagate detected transformed type to the output pin");
                            let inner = detect.to_ty_or_hint();
                            let array = HintedPrimitiveType::Array(Box::new(inner.clone()));
                            let result = HintedPrimitiveType::Result(Box::new((inner, array)));
                            is_progress |=
                                AssignedType::match_or_write(result, &mut outs[0]).is_progress();

                            Self { pins, is_progress }
                        }
                        ExpectSome { .. } => {
                            // All inputs should be the same type, and the output is the same type.
                            let (ins, outs) = Self::pin_io_slices_mut(pins, node);
                            assert_eq!(outs.len(), 1);

                            // Get a known type in any of the pins.
                            trace!("ExpectSome: get a known type in any of the pins");
                            let mut detect = outs[0].clone();
                            for i in ins.iter_mut() {
                                is_progress |= AssignedType::union(&mut detect, i).is_progress();
                            }

                            trace!("Propagate detected type to all pins: {detect:#?}");
                            for i in ins.iter_mut() {
                                is_progress |= AssignedType::union(&mut detect, i).is_progress();
                            }
                            is_progress |=
                                AssignedType::union(&mut detect, &mut outs[0]).is_progress();

                            Self { pins, is_progress }
                        }
                        Match {
                            values, wildcard, ..
                        } => {
                            // TODO where do we validate this node for variant correctness?
                            // Do we do this already? :)

                            trace!("Match: resolve types of input and output pins");

                            let input_val = {
                                if let Some((pat, _)) = values.first() {
                                    Some(pat.value())
                                } else {
                                    None
                                }
                            };

                            let output_val = {
                                if let Some((_, value)) = values.first() {
                                    Some(value)
                                } else if let Some(wildcard) = wildcard.as_ref() {
                                    Some(wildcard)
                                } else {
                                    None
                                }
                            };

                            let (ins, outs) = Self::pin_io_slices_mut(pins, node);

                            trace!("Set input pin types");
                            if let Some(input_ty_iter) = input_val.map(Value::iter) {
                                for (ty, val) in ins.iter_mut().zip(input_ty_iter) {
                                    is_progress |=
                                        AssignedType::match_or_write(val.type_of().into(), ty)
                                            .is_progress();
                                }
                            }

                            trace!("Set output pin types");
                            if let Some(output_ty_iter) = output_val.map(Value::iter) {
                                for (ty, val) in outs.iter_mut().zip(output_ty_iter) {
                                    is_progress |=
                                        AssignedType::match_or_write(val.type_of().into(), ty)
                                            .is_progress();
                                }
                            }

                            Self { pins, is_progress }
                        }
                        Func(_) => {
                            trace!("Func: unresolvable within local resolver");
                            Self { pins, is_progress }
                        }
                        Constant(value) => {
                            let ty = value.type_of().into();
                            assert_eq!(pins.len(), 1);
                            is_progress |=
                                AssignedType::match_or_write(ty, &mut pins[0]).is_progress();

                            Self { pins, is_progress }
                        }
                        _ => unreachable!("unhandled dynamic IO relation"),
                    }
                }
                IoRelation::Defined => {
                    // All pins known from constant scope.
                    let inputs = node.static_inputs().unwrap();
                    let outputs = node.static_outputs().unwrap();

                    let (ins, outs) = Self::pin_io_slices_mut(pins, node);
                    assert_eq!(ins.len(), inputs.len());
                    assert_eq!(outs.len(), outputs.len());

                    for (const_input, target_input) in inputs.iter().copied().zip(ins.iter_mut()) {
                        let set_to = const_input.into();
                        is_progress |=
                            AssignedType::match_or_write(set_to, target_input).is_progress();
                    }

                    for (const_output, target_output) in
                        outputs.iter().copied().zip(outs.iter_mut())
                    {
                        let set_to = const_output.into();
                        is_progress |=
                            AssignedType::match_or_write(set_to, target_output).is_progress();
                    }

                    Self { pins, is_progress }
                }
                IoRelation::ConstCount => {
                    use NodeStub::*;

                    match node {
                        Drop | Output { .. } | Crash { .. } | Unreachable { .. } => {
                            // Unresolvable without external information.
                            trace!("Is not a resolvable node, skipping: {node:?}");
                            Self { pins, is_progress }
                        }
                        ExpectSome { .. } => {
                            let const_input = *node.static_inputs().unwrap().first().unwrap();
                            let (ins, outs) = Self::pin_io_slices_mut(pins, node);
                            assert_eq!(ins.len(), 1);
                            assert_eq!(outs.len(), 1);

                            trace!(
                                "ExpectSome: resolve inner type of input Option or type of output"
                            );
                            let inner = {
                                let input = ins.first_mut().unwrap();
                                let output = outs.first_mut().unwrap();

                                if let Some(ty) = input.opt_ty() {
                                    if let HintedPrimitiveType::Option(ty) = ty {
                                        let mut v = AssignedType::with((**ty).to_owned());
                                        is_progress |=
                                            AssignedType::union(output, &mut v).is_progress();
                                    } else {
                                        trace!("Input is not an Option, but expected one");
                                        output.mark_err();
                                    }
                                }

                                output.to_owned()
                            };

                            if let Some(inner) = inner.opt_ty() {
                                trace!("Inner type is resolved as: {inner:#?}");
                                is_progress |= AssignedType::match_or_write(
                                    HintedPrimitiveType::Option(Box::new(inner.to_owned())),
                                    &mut outs[0],
                                )
                                .is_progress();

                                is_progress |=
                                    AssignedType::match_or_write(const_input.into(), &mut ins[0])
                                        .is_progress();
                            }

                            Self { pins, is_progress }
                        }
                        Constant(v) => {
                            let ty = v.type_of().into();
                            let (ins, outs) = Self::pin_io_slices_mut(pins, node);
                            assert_eq!(ins.len(), 0);
                            assert_eq!(outs.len(), 1);

                            is_progress |=
                                AssignedType::match_or_write(ty, &mut outs[0]).is_progress();

                            Self { pins, is_progress }
                        }
                        OkOrCrash { .. } => {
                            let (ins, outs) = Self::pin_io_slices_mut(pins, node);
                            assert_eq!(ins.len(), 1);
                            assert_eq!(outs.len(), 1);

                            trace!(
                                "OkOrCrash: resolve inner type of input Result or type of output"
                            );
                            let mut ty = outs[0].clone();
                            if let Some(out) = ty.opt_ty() {
                                let mut input = AssignedType::with(HintedPrimitiveType::Result(
                                    Box::new((out.to_owned(), HintedPrimitiveType::Hint)),
                                ));
                                is_progress |=
                                    AssignedType::union(&mut ty, &mut input).is_progress();
                            }
                            if let Some(inp) = ins[0].opt_ty() {
                                if let HintedPrimitiveType::Result(boxed) = inp {
                                    let (ok, _) = &**boxed;
                                    is_progress |=
                                        AssignedType::match_or_write(ok.to_owned(), &mut ty)
                                            .is_progress();
                                }
                            }
                            trace!("OkOrCrash: resolved type: {:#?}", outs[0].opt_ty());

                            Self { pins, is_progress }
                        }
                        _ => unreachable!("unhandled const count relation"),
                    }
                }
            }
        };

        trace!("Resolved pins: {:#?}", result.pins);
        Ok(result)
    }

    /// Prefill missing types per static information.
    ///
    /// # Panics
    /// Pin count should be correct at this point. If it is not, this function will panic.
    fn prefill_with_static(
        node: &NodeStub,
        pins: &mut Vec<AssignedType>,
    ) -> Result<bool, PinResolutionError> {
        let mut is_progress = false;

        macro_rules! iterate {
            ($kind:ident) => {
                if let Some($kind) = node.$kind() {
                    for (i, t) in $kind.iter().copied().enumerate() {
                        use TypePrecision::*;
                        match pins[i].union_type_with(t.into()) {
                            Same | Uncertain => {}
                            Precise => is_progress = true,
                            Incompatible => {
                                info!("incompatible types for pin {i}");
                                // Moving forward, union_type_with marked this
                                // pin as erroneous, which may affect
                                // some resolution quality, but still we're
                                // doing best-effort.

                                // We still mark this as progress, as we did
                                // change the state of the pin.
                                is_progress = true;
                            }
                        }
                    }
                }
            };
        }

        trace!("static inputs prefill");
        iterate!(static_inputs);

        trace!("static outputs prefill");
        iterate!(static_outputs);

        trace!("static prefill progress = {is_progress}");
        Ok(is_progress)
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

/// Error that can happen when resolving pin types for the node.
#[derive(Debug)]
pub enum PinResolutionError {
    /// Preallocated buffer to represent node pins does not match actual node
    /// pin count.
    PinNumberMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    macro_rules! edge {
        ($from_node:expr, $from_pin:expr => $to_node:expr, $to_pin:expr) => {
            Edge {
                from: OutputPin(Pin::new($from_node, $from_pin)),
                to: InputPin(Pin::new($to_node, $to_pin)),
            }
        };
    }

    #[test]
    fn resolve0() {
        crate::tests::init();

        let mut canvas = Canvas::new();
        let file = canvas.add_node(NodeStub::File, ());
        let in0 = canvas.add_node(NodeStub::Input { valid_names: smallvec!["a".into()] }, ());
        let in1 = canvas.add_node(NodeStub::Input{ valid_names: smallvec!["b".into()] }, ());
        let strop = canvas.add_node(NodeStub::StrOp(StrOp::Lowercase), ());
        let drop0 = canvas.add_node(NodeStub::Drop, ());
        let drop1 = canvas.add_node(NodeStub::Drop, ());

        canvas.add_edge(edge!(file, 0 => in0, 0)).unwrap();
        canvas.add_edge(edge!(file, 0 => in1, 0)).unwrap();
        canvas.add_edge(edge!(in0, 1 => strop, 0)).unwrap();
        canvas.add_edge(edge!(in1, 1 => drop0, 0)).unwrap();
        canvas.add_edge(edge!(strop, 1 => drop1, 0)).unwrap();

        let ty = canvas.calc_pin_type(Pin::new(drop0, 0)).unwrap().unwrap();
        assert_eq!(ty, PrimitiveType::Str);
    }
}
