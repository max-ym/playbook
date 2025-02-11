use std::{borrow::Borrow, collections::VecDeque};

use super::*;

use hashbrown::HashSet;
use log::{debug, error, info, trace};
use smallvec::SmallVec;

#[derive(Debug, Clone)]
struct AssignedType {
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

    /// Set of nodes with modifications that require revalidation.
    nodes_mod: HashSet<Id>,

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
#[derive(Debug)]
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
    pub fn calc_pin_type(&mut self, pin: Pin) -> Result<Option<&PrimitiveType>, PinQueryError> {
        let ty = Validator::resolve_pin_type(self, pin)?;
        Ok(ty.ty[pin.order as usize].ty.as_ref())
    }
}

impl Validator {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            nodes_mod: HashSet::new(),
            nodes: Vec::new(),
            pin_groups: Vec::new(),
        }
    }

    pub fn alloc(&mut self, capacity: usize) {
        if self.buf.capacity() == 0 {
            self.buf.reserve(64);
        }
        if self.nodes_mod.capacity() == 0 {
            self.nodes_mod.reserve(capacity);
        }
        if self.nodes.capacity() == 0 {
            self.nodes.reserve(capacity);
        }
        if self.pin_groups.capacity() == 0 {
            self.pin_groups.reserve(capacity);
        }
    }

    fn node(&self, id: Id) -> Option<&NodeData> {
        self.nodes
            .binary_search_by(|n| n.id.cmp(&id))
            .ok()
            .map(|idx| &self.nodes[idx])
    }

    /// Peek into the pin type. If it is not known, return [None].
    pub fn peek_pin_type<T>(
        canvas: &Canvas<T>,
        pin: Pin,
    ) -> Result<Option<&HintedPrimitiveType>, PinQueryError> {
        let is_actual = !canvas.valid.nodes_mod.contains(&pin.node_id);
        if !is_actual {
            return Ok(None);
        }

        let node = canvas
            .valid
            .node(pin.node_id)
            .ok_or(PinQueryError::NodeNotFound(pin.node_id))?;
        let pin_group_idx = node.ty[pin.order as usize];
        let pin_group = &canvas.valid.pin_groups[pin_group_idx];
        Ok(pin_group.ty.as_ref().map(|v| v.as_ref()).unwrap_or(None))
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

    pub fn resolve_pin_type<T>(
        canvas: &mut Canvas<T>,
        pin: Pin,
    ) -> Result<&NodeData, PinQueryError> {
        if !canvas.valid.nodes_mod.contains(&pin.node_id) {
            if let Some(node) = canvas.valid.node(pin.node_id) {
                trace!("returning already resolved node");
                return Ok(node);
            }
        }

        // Next node to resolve.
        let mut resolve_next = VecDeque::with_capacity(64);
        resolve_next.push_front(
            canvas
                .node_id_to_idx(pin.node_id)
                .ok_or(PinQueryError::NodeNotFound(pin.node_id))?,
        );

        while !resolve_next.is_empty() {
            // let node = &canvas.nodes[node_idx];
            // if node.is_resolved() {
            //     continue;
            // }

            Self::resolve_node(canvas, &mut resolve_next);
        }

        Ok(canvas
            .valid
            .node(pin.node_id)
            .expect("should exist due to ran validation"))
    }

    /// Resolve types for the pins of a node comming first in the queue. Old resolution, if
    /// present, will be discarded. The function can again place this node into `resolve_next`
    /// queue if the resolution was not complete, but is believed to proceed once other nodes
    /// in the queue are processed.
    fn resolve_node<T>(canvas: &mut Canvas<T>, resolve_next: &mut VecDeque<NodeIdx>) {
        use std::convert::Infallible as Never;

        let node_idx = resolve_next.pop_back().unwrap() as usize;
        let node_id = canvas.nodes[node_idx].id;
        let buf_token = Validator::load_buf_for(canvas, node_id);

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
                        continue;
                    } else {
                        unreachable!(
                            "node should be resolved or progress should be made to be `Ok`"
                        );
                    }
                }
                Err(PinNumberMismatch) => {
                    unreachable!("pin number mismatch, invalid preallocation?");
                    // Pins should have been preallocated per node requirements, and
                    // if we reach this point, it means that particular code
                    // likely has a bug.
                }
                Err(UnionConflict) => {
                    info!("union conflict for node {node_idx}");
                    // We would not be able to resolve this node further, so we stop here.
                    break;
                }
                Err(RemainingUnknownPins) => {
                    trace!("remaining unknown pins for node {node_idx}");
                    // No progress was made, so we end with this node.
                    break;
                }
            };
        }

        Validator::save_buf_for(canvas, buf_token);
        Validator::neighbors_scan_for_revalidation(node_idx, canvas, resolve_next);
        todo!()
    }

    /// Check whether neighbors of the node connected by the edges require revalidation, and if
    /// so, add them to the `resolve_next` queue.
    ///
    /// True is returned if at least one neighbor requires revalidation and was added to the queue.
    fn neighbors_scan_for_revalidation<T>(
        node_idx: usize,
        canvas: &mut Canvas<T>,
        resolve_next: &mut VecDeque<NodeIdx>,
    ) -> bool {
        let node_id = canvas.nodes[node_idx].id;
        let mut require_others = false;
        for edge in canvas
            .node_edge_io_iter(node_id)
            .expect("existence proved by caller")
        {
            let other_node_id = edge.oppose(node_id).node_id;
            if canvas.valid.nodes_mod.contains(&other_node_id) {
                require_others = true;
                resolve_next.push_front(
                    canvas
                        .node_id_to_idx(other_node_id)
                        .expect("correct edge maintenance on add/remove operations"),
                );
            }
        }
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
            buf[other.order as usize].ty = ty.cloned();
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

/// Release token to remind to save or discard the buffer.
#[derive(Debug)]
struct BufferReleaseToken(Id);

impl BufferReleaseToken {
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
            return Ok(Self {
                pins,
                is_progress: false,
            });
        }

        let mut is_progress = ResolvePinTypes::prefill_with_static(node, pins)?;

        let result = 'result: {
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
                            if let (Some(i), Some(o)) = (ins.first_mut(), outs.last_mut()) {
                                is_progress |= AssignedType::union(i, o).is_progress();
                            }

                            // #2
                            let mut opt = AssignedType::with(HintedPrimitiveType::Option(
                                Box::new(HintedPrimitiveType::Unit),
                            ));
                            let mut local_progress = false;
                            for _ in 0..=1 {
                                for i in ins.iter_mut().skip(1) {
                                    local_progress |=
                                        AssignedType::union(i, &mut opt).is_progress();
                                }
                                if !local_progress {
                                    // Second iteration won't propagate any new information.
                                    break;
                                }
                            }
                            is_progress |= local_progress;

                            let is_first_opt = ins.first().map(|i| {
                                matches!(i.opt_ty(), Some(HintedPrimitiveType::Option(_)))
                            });
                            let is_out_opt = outs.last().map(|o| {
                                matches!(o.opt_ty(), Some(HintedPrimitiveType::Option(_)))
                            });

                            // #3
                            let mut inner = 'inner: {
                                // At this point, "opt" variable has high chance of being
                                // already resolved.
                                if opt.is_resolved() {
                                    // Unwrap from option and return.
                                    if let HintedPrimitiveType::Option(inner) = opt.to_ty_or_hint()
                                    {
                                        break 'inner AssignedType::with(*inner);
                                    } else {
                                        unreachable!(
                                            "resolved type should be Option in this position"
                                        );
                                    }
                                }

                                // If the first input is not an Option, we can use it as the inner type.
                                if let Some(false) = is_first_opt {
                                    if let Some(ty) = ins.first().and_then(|i| i.opt_ty()) {
                                        break 'inner AssignedType::with(*ty);
                                    }
                                }
                                // Try to use the output's optional type's inner type.
                                if let Some(false) = is_out_opt {
                                    if let Some(ty) = outs.last().and_then(|o| o.opt_ty()) {
                                        break 'inner AssignedType::with(*ty);
                                    }
                                }

                                AssignedType::default()
                            };

                            // Try to give more precise information about the inner type to
                            // our Option-holding variable.
                            if let Some(inner) = inner.opt_ty() {
                                is_progress |= AssignedType::union(
                                    &mut AssignedType::with(HintedPrimitiveType::Option(Box::new(
                                        inner.to_owned(),
                                    ))),
                                    &mut opt,
                                )
                                .is_progress();
                            }

                            // Propagate the inner type (without a wrapper when applicable) to all pins.

                            // Repeat option union as our option should now be the best guess.
                            for i in ins.iter_mut().skip(1) {
                                is_progress |= AssignedType::union(i, &mut opt).is_progress();
                            }
                            // Set the correct first pin, which may or may not be option.
                            match is_first_opt {
                                Some(true) => {
                                    is_progress |=
                                        AssignedType::union(ins.first_mut().unwrap(), &mut opt)
                                            .is_progress();
                                }
                                Some(false) => {
                                    is_progress |=
                                        AssignedType::union(ins.first_mut().unwrap(), &mut inner)
                                            .is_progress();
                                }
                                None => {
                                    // Do nothing, as we don't know the first pin type for sure,
                                    // it could be either Option or not.
                                }
                            }
                            match is_out_opt.or(is_first_opt) {
                                Some(true) => {
                                    is_progress |=
                                        AssignedType::union(&mut outs[0], &mut opt)
                                            .is_progress();
                                }
                                Some(false) => {
                                    is_progress |=
                                        AssignedType::union(&mut outs[0], &mut inner)
                                            .is_progress();
                                }
                                None => {
                                    // Do nothing, as we don't know the output pin type for sure,
                                    // it could be either Option or not.
                                }
                            }

                            Self { pins, is_progress }
                        }
                        ExpectOne { .. } => {
                            // All inputs should be the same type, and the output is the result:
                            // Result<type, Array<type>>.

                            let (ins, outs) = Self::pin_io_slices_mut(pins, node);
                            assert_eq!(outs.len(), 1);

                            // Get a known type in any of the input pins.
                            let detect = {
                                let mut detect = None;
                                for i in ins.iter() {
                                    if let Some(i) = i {
                                        detect = Some(i);
                                        break;
                                    }
                                }
                                detect
                            };

                            if let Some(detect) = detect.map(ToOwned::to_owned) {
                                for i in ins.iter_mut() {
                                    is_progress |=
                                        ResolvePinTypes::match_types_write(detect.clone(), i)?;
                                }

                                let out = PrimitiveType::Result(Box::new((
                                    detect.clone(),
                                    PrimitiveType::Array(Box::new(detect)),
                                )));

                                is_progress |=
                                    ResolvePinTypes::match_types_write(out, &mut outs[0])?;
                            }

                            Self { pins, is_progress }
                        }
                        ExpectSome { .. } => {
                            // All inputs should be the same type, and the output is the same type.
                            let (ins, outs) = Self::pin_io_slices_mut(pins, node);
                            assert_eq!(outs.len(), 1);

                            // Get a known type in any of the pins.
                            let detect = {
                                let mut detect = None;
                                for i in ins.iter().chain(outs.iter()) {
                                    if let Some(i) = i {
                                        detect = Some(i);
                                        break;
                                    }
                                }
                                detect
                            };

                            if let Some(detect) = detect.map(ToOwned::to_owned) {
                                for i in ins.iter_mut() {
                                    is_progress |=
                                        ResolvePinTypes::match_types_write(detect.clone(), i)?;
                                }

                                is_progress |=
                                    ResolvePinTypes::match_types_write(detect, &mut outs[0])?;
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
                IoRelation::ConstCount => {
                    use NodeStub::*;
                    todo!()
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
                        if let Some(t) = t.map(Into::into) {
                            use TypePrecision::*;
                            match pins[i].union_type_with(t) {
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
