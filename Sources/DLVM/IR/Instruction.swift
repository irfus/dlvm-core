//
//  Instruction.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

import DLVMTensor

public enum MemoryLocation {
    case host, compute
}

public enum InstructionKind {
    /** Control flow **/
    /// Unconditionally branch to basic block
    case branch(BasicBlock, [Use])
    /// Conditional branch depending on the value
    case conditional(Use, BasicBlock, [Use], BasicBlock, [Use])
    /// Return
    case `return`(Use?)

    /** Tensor operations **/
    /// - Note: Requires compute function context when applied to 
    /// non-scalar tensor buffers
    /// Monomorphic unary operation
    case unary(UnaryOp, Use)
    /// Monomorphic binary operation
    case binary(BinaryOp, Use, Use)
    /// Scan operation with optional axis
    /// If axis is not given, scan is performed on contiguous elements
    case scan(AssociativeOp, Use, axis: Int?)
    /// Reduction operation with optional axis
    /// If axis is not given, reduction is performed on contiguous elements
    case reduce(AssociativeOp, Use, axis: Int?)
    /// Matrix multiplication operation
    case matrixMultiply(Use, Use)
    /// Concatenation operation
    case concatenate([Use], axis: Int)
    /// Transpose
    case transpose(Use)
    /// Data type cast operation
    case dataTypeCast(Use, DataType)

    /** Cost-free casts **/
    /// Shape cast operation
    case shapeCast(Use, TensorShape)
    /// Bitcast
    case bitCast(Use, Type)

    /** Aggregate operation **/
    /// Extract an element from tensor, tuple, or array
    case extract(from: Use, at: [Int])
    /// Insert an element to tensor, tuple, or array
    case insert(Use, to: Use, at: [Int])

    /** Function invocation **/
    /// Function application
    case apply(Use, [Use])
    /// Gradient application, using implicit automatic differentiation
    /// Will be replaced by `compute` or `apply` through GradientExpansion
    /// transform pass
    case applyGradient(Use, [Use])
    /// Compute function launcher
    case compute(Use, [Use], in: Use)
    /// Compute gradient function launcher
    /// Will be replaced by `compute` or
    case computeGradient(Use, [Use], in: Use)

    /** Heap memory of host and device **/
    /** Memory **/
    /// Allocate host stack memory, returning a pointer
    case allocateStack(Type, Use) /// => +*T
    case allocateHeapRaw(Type, MemoryLocation, count: Use) /// => +*T
    /// Allocate raw heap memory, returning a pointer
    case allocateHeap(Type, MemoryLocation) /// => +&T
    case referenceAddress(Use) /// &T => *T; +&T => +*T
    /// Reference-counted box
    case allocateBox(Type, MemoryLocation) /// => box<T>
    case projectBox(Use) /// @box T => +&T
    /// Retain/release a box via reference counter
    case retain(Use)
    case release(Use)
    /// Dealloc any heap memory 
    case deallocate(Use)
    /// Load value from pointer on the host
    case load(Use)
    /// Store value to pointer on the host
    case store(Use, to: Use)
    /// GEP
    case elementPointer(Use, [Use])
    /// Memory copy
    case copy(from: Use, to: Use, count: Use)
    /// Trap
    case trap
}

public final class Instruction : IRSubUnit, Value, Definition, MaybeNamed {
    public typealias Parent = BasicBlock
    public var name: String?
    public var kind: InstructionKind
    public unowned var parent: BasicBlock
    public internal(set) var analysisManager: AnalysisManager<Instruction> = AnalysisManager()
    public internal(set) var transformManager: TransformManager<Instruction> = TransformManager()

    public required init(name: String?, kind: InstructionKind, parent: BasicBlock) {
        self.name = name
        self.kind = kind
        self.parent = parent
    }
    
    public var type: Type {
        return kind.type
    }

    public func makeUse() -> Use {
        return .instruction(type, self)
    }
}

// MARK: - Predicates
public extension InstructionKind {
    var isTerminator: Bool {
        switch self {
        case .branch, .conditional, .`return`:
            return true
        default:
            return false
        }
    }

    var isReturn: Bool {
        switch self {
        case .`return`: return true
        default: return false
        }
    }

    var accessesMemory: Bool {
        switch self {
        case .allocateStack, .allocateHeapRaw, .allocateBox,
             .projectBox, .load, .store, .deallocate:
            return true
        default:
            return false
        }
    }

    var isComputeOnly: Bool {
        switch self {
        case .reduce, .scan, .matrixMultiply, .concatenate, .transpose: return true
        default: return false
        }
    }
}

infix operator <>

extension TensorShape {
    public static func <> (lhs: TensorShape, rhs: TensorShape) -> TensorShape? {
        return lhs.mutuallyBroadcasted(with: rhs)
    }
}

extension InstructionKind {

    public var type: Type {
        switch self {
        case let .binary(.associative(.arithmetic), v1, v2):
            switch (v1.type.unaliased, v2.type.unaliased) {
            /// Non-compute scalar
            case let (.tensor([], t1), .tensor([], t2)) where t1 == t2:
                return .tensor([], t1)
            /// Compute tensor
            case let (.pointer(.tensor(s1, t1), .compute, .mutable),
                      .pointer(.tensor(s2, t2), .compute, .mutable)) where t1 == t2:
                return (s1 <> s2).flatMap { .pointer(.tensor($0, t1), .compute, .mutable) }
                    ?? .invalid
            default:
                return .invalid
            }

        case let .binary(.associative(.boolean), v1, v2),
             let .binary(.comparison, v1, v2):
            switch (v1.type.unaliased, v2.type.unaliased) {
            /// Non-compute scalar
            case let (.tensor([], t1), .tensor([], t2)) where t1 == t2:
                return .tensor([], .bool)
            /// Compute tensor
            case let (.pointer(.tensor(s1, t1), .compute, .mutable),
                      .pointer(.tensor(s2, t2), .compute, .mutable)) where t1 == t2:
                return (s1 <> s2).flatMap { .pointer(.tensor($0, .bool), .compute, .mutable) }
                    ?? .invalid
            default:
                return .invalid
            }

        /// Compute-only
        case let .matrixMultiply(v1, v2):
            guard case let .pointer(.tensor(s1, t1), .compute, .mutable) = v1.type.unaliased,
                  case let .pointer(.tensor(s2, t2), .compute, .mutable) = v2.type.unaliased,
                  let newShape = s1.matrixMultiplied(with: s2),
                  t1 == t2 else { return .invalid }
            return .pointer(.tensor(newShape, t1), .compute, .mutable)

        case let .unary(_, v1):
            switch v1.type.unaliased {
            /// Non-compute scalar
            case .tensor([], _):
                return v1.type
            /// Compute tensor
            case let .pointer(.tensor(s, dt), .compute, .mutable):
                return v1.type
            default:
                return .invalid
            }

        /// Compute-only
        case let .reduce(op, v1, nil):
            switch (op, v1.type.unaliased) {
            case let (.arithmetic, .pointer(.tensor(s1, t1), .compute, .mutable)) where t1.isNumeric:
                return .pointer(.tensor([], t1), .compute, .mutable)
            case let (.boolean, .pointer(.tensor(s1, .bool), .compute, .mutable)):
                return .pointer(.tensor([], .bool), .compute, .mutable)
            default:
                return .invalid
            }

        /// Compute-only
        case let .reduce(op, v1, axis: axis?):
            switch (op, v1.type.unaliased) {
            case let (.arithmetic, .pointer(.tensor(s1, t1), .compute, .mutable))
                    where t1.isNumeric && axis < s1.rank:
                return .pointer(.tensor(s1.droppingDimension(axis), t1), .compute, .mutable)
            case let (.boolean, .pointer(.tensor(s1, .bool), .compute, .mutable))
                    where axis < s1.rank:
                return .pointer(.tensor(s1.droppingDimension(axis), .bool), .compute, .mutable)
            default:
                return .invalid
            }

        /// Compute-only
        case let .scan(op, v1, _):
            guard case .pointer(.tensor, .compute, .mutable) = v1.type.unaliased else {
                return .invalid
            }
            return v1.type

        /// Compute-only
        case let .concatenate(vv, axis):
            guard let first = vv.first,
                  case let .pointer(.tensor(s1, t1), .compute, .mutable) = first.type.unaliased
                else { return .invalid }
            var accShape: TensorShape = s1
            for v in vv.dropFirst() {
                guard case let .pointer(.tensor(shape, type), .compute, .mutable) = v.type.unaliased,
                      type == t1,
                      let newShape = accShape.concatenating(with: shape, alongDimension: axis)
                    else { return .invalid }
                accShape = newShape
            }
            return .pointer(.tensor(accShape, t1), .compute, .mutable)

        /// Compute-only
        case let .transpose(v1):
            guard case let .pointer(.tensor(s1, t1), .compute, .mutable) = v1.type.unaliased
                else { return .invalid }
            return .pointer(.tensor(s1.transpose, t1), .compute, .mutable)
            
        case let .dataTypeCast(v1, dt):
            switch v1.type.unaliased {
            case let .tensor([], t1) where t1.canCast(to: dt):
                return .tensor([], dt)
            case let .pointer(.tensor(s1, t1), .compute, .mutable) where t1.canCast(to: dt):
                return .pointer(.tensor(s1, dt), .compute, .mutable)
            default:
                return .invalid
            }
            
        case let .shapeCast(v1, s):
            switch v1.type.unaliased {
            case let .tensor(s1, t1) where s1.contiguousSize == s.contiguousSize:
                return .tensor(s, t1)
            case let .pointer(.tensor(s1, t1), .compute, .mutable)
                    where s1.contiguousSize == s.contiguousSize:
                return .pointer(.tensor(s, t1), .compute, .mutable)
            default: return .invalid
            }

        case let .apply(f, vv), let .applyGradient(f, vv):
            switch f.type.unaliased {
            case let .function(actual, ret),
                 let .pointer(.function(actual, ret), _, _):
                guard actual == vv.map({$0.type}) else { fallthrough }
                return ret
            default:
                return .invalid
            }

        case let .applyGradient(f, vv), let .computeGradient(f, vv, in: _):
            let actual = vv.map{$0.type}
            switch f {
            case let .function(ty, funDef)
                where ty == funDef.type
                   && funDef.isDifferentiable
                   && funDef.acceptsArguments(actual):
                return funDef.result
            default:
                return .invalid
            }

        case let .extract(from: v, at: indices):
            return v.type.subtype(at: indices) ?? .invalid

        case let .insert(src, to: dest, at: indices):
            guard let subtype = dest.type.subtype(at: indices), subtype == src.type else {
                return .invalid
            }
            return dest.type

        case let .allocateStack(type, _):
            return .reference(type, .host, .mutable)

        case let .load(v):
            guard case let .pointer(t, .host, _) = v.type.unaliased else { return .invalid }
            return t

        case let .elementPointer(v, ii):
            guard case .pointer = v.type else { return .invalid }
            func gepType(type: Type, indices: ArraySlice<Use>) -> Type {
                guard let idx = indices.first else { return type }
                switch type.unaliased {
                case let .pointer(t, loc, mut):
                    return .pointer(gepType(type: t, indices: indices.dropFirst()), loc, mut)
                case let .tensor(shape, dt):
                    return .tensor(shape.dropFirst(), dt)
                case let .array(t, _):
                    return gepType(type: t, indices: indices.dropFirst())
                case let .tuple(tt):
                    /// Current index must be an int literal
                    guard case let .literal(_, litVal) = idx,
                          case let .scalar(.int(i)) = litVal.literal else {
                        return .invalid
                    }
                    return gepType(type: tt[i], indices: indices.dropFirst())
                default:
                    return .invalid
                }
            }
            return gepType(type: v.type, indices: ArraySlice(ii))

        case let .compute(v, args, in: env):
            let actual = args.map{$0.type}
            switch (v, env.type) {
            case let (.function(_, f1), .computeGraph(f2))
                where f1 === f2 && f1.acceptsArguments(actual):
                return f1.result
            default:
                return .invalid
            }

        case let .bitCast(_, t):
//            guard v.type.size == t.size else { return .invalid }
            return t

        case let .allocateBox(t, loc):
            return .box(t, loc)

        case let .allocateHeapRaw(t, loc, count: _):
            return .pointer(t, loc, .mutable)

        case let .referenceAddress(v):
            guard case let .reference(t, loc, mut) = v.type.unaliased else { return .invalid }
            return .pointer(t, loc, mut)

        case let .projectBox(v):
            guard case let .box(t, loc) = v.type else { return .invalid }
            return .reference(t, loc, .mutable)

        case let .allocateHeap(t, loc):
            return .reference(t, loc, .mutable)

        case .store, .copy, .deallocate,
             .branch, .conditional, .return, .retain, .release, .trap:
            return .void
        }
    }

    public static var scope: Scope = .local

}

extension Instruction : User {
    public var operands: [Use] {
        switch kind {
        case let .binary(_, op1, op2),
             let .matrixMultiply(op1, op2),
             let .insert(op1, to: op2, at: _):
            return [op1, op2]
        case let .unary(_, op), let .reduce(_, op, _), let .scan(_, op, _), let .transpose(op),
             let .shapeCast(op, _), let .dataTypeCast(op, _), let .bitCast(op, _), let .return(op?),
             let .extract(from: op, at: _),
             let .store(op, _), let .load(op),
             let .elementPointer(op, _),
             let .deallocate(op),
             let .allocateStack(_, op), let .allocateHeapRaw(_, _, count: op),
             let .projectBox(op), let .referenceAddress(op),
             let .release(op), let .retain(op):
            return [op]
        case .concatenate(let ops, _),
             .branch(_, let ops):
            return ops
        case let .conditional(cond, _, thenArgs, _, elseArgs):
            return [cond] + thenArgs + elseArgs
        case let .apply(f, args), let .applyGradient(f, args):
            return [f] + args
        case let .compute(f, args, in: env), let .computeGradient(f, args, in: env):
            return [f] + args + [env]
        case let .copy(from: op1, to: op2, count: op3):
            return [op1, op2, op3]
        case .return(nil), .allocateBox, .trap, .allocateHeap:
            return []
        }
    }
}

public extension Instruction {
    func substitute(_ newUse: Use, for use: Use) {
        kind = kind.substituting(newUse, for: use)
    }
}

public extension InstructionKind {
    /// Substitutes new use for old use
    /// - Note: The current implementation is a vanilla tedious switch 
    /// matching all the permutations (a.k.a. very bad).
    func substituting(_ new: Use, for old: Use) -> InstructionKind {
        let condSubst = {$0 == old ? new : $0}
        switch self {
        case .branch(let dest, let args):
            return .branch(dest, args.map(condSubst))
        case let .conditional(cond, thenBB, thenArgs, elseBB, elseArgs):
            let newCond = cond == old ? new : cond
            return .conditional(newCond,
                                thenBB, thenArgs.map(condSubst),
                                elseBB, elseArgs.map(condSubst))
        case .return(old?):
            return .return(new)
        case .unary(let fun, old):
            return .unary(fun, new)
        case .binary(let fun, old, old):
            return .binary(fun, new, new)
        case .binary(let fun, old, let use2):
            return .binary(fun, new, use2)
        case .binary(let fun, let use1, old):
            return .binary(fun, use1, new)
        case let .concatenate(uses, axis: axis):
            return .concatenate(uses.map(condSubst), axis: axis)
        case .transpose(old):
            return .transpose(new)
        case .reduce(let fun, old, axis: let axis):
            return .reduce(fun, new, axis: axis)
        case .matrixMultiply(old, let use2):
            return .matrixMultiply(new, use2)
        case .matrixMultiply(let use1, old):
            return .matrixMultiply(use1, new)
        case .matrixMultiply(old, old):
            return .matrixMultiply(new, new)
        case .shapeCast(old, let shape):
            return .shapeCast(new, shape)
        case .dataTypeCast(old, let type):
            return .dataTypeCast(new, type)
        case .compute(old, let uses, in: old):
            return .compute(new, uses.map(condSubst), in: new)
        case .compute(old, let uses, in: let v2):
            return .compute(old, uses.map(condSubst), in: v2)
        case .compute(let v1, let uses, in: old):
            return .compute(v1, uses.map(condSubst), in: new)
        case .compute(let v1, let uses, in: let v2):
            return .compute(v1, uses.map(condSubst), in: v2)
        case .computeGradient(old, let uses, old):
            return .computeGradient(new, uses.map(condSubst), in: new)
        case .computeGradient(old, let uses, in: let v2):
            return .computeGradient(old, uses.map(condSubst), in: v2)
        case .computeGradient(let v1, let uses, in: old):
            return .computeGradient(v1, uses.map(condSubst), in: new)
        case let .apply(f, uses):
            return .applyGradient(f, uses.map(condSubst))
        case let .applyGradient(f, uses):
            return .applyGradient(f, uses.map(condSubst))
        case .extract(from: old, at: let i):
            return .extract(from: new, at: i)
        case .insert(old, to: old, at: let indices):
            return .insert(new, to: new, at: indices)
        case .insert(old, to: let use1, at: let indices):
            return .insert(new, to: use1, at: indices)
        case .insert(let use1, to: old, at: let indices):
            return .insert(use1, to: new, at: indices)
        case .bitCast(old, let targetT):
            return .bitCast(new, targetT)
        case .elementPointer(old, let indices):
            return .elementPointer(new, indices)
        case .store(old, to: let dest):
            return .store(new, to: dest)
        case .store(let val, to: old):
            return .store(val, to: new)
        case .load(old):
            return .load(new)
        case .allocateStack(let ty, old):
            return .allocateStack(ty, new)
        case .allocateHeapRaw(let ty, let memLoc, count: old):
            return .allocateHeapRaw(ty, memLoc, count: new)
        case .deallocate(old):
            return .deallocate(new)
        case .copy(from: old, to: old, count: old):
            return .copy(from: new, to: new, count: new)
        case .copy(from: old, to: old, count: let v3):
            return .copy(from: new, to: new, count: v3)
        case .copy(from: old, to: let v2, count: old):
            return .copy(from: new, to: v2, count: new)
        case .copy(from: old, to: let v2, count: let v3):
            return .copy(from: new, to: v2, count: v3)
        case .copy(from: let v1, to: old, count: old):
            return .copy(from: v1, to: new, count: new)
        case .copy(from: let v1, to: old, count: let v3):
            return .copy(from: v1, to: new, count: v3)
        case .copy(from: let v1, to: let v2, count: old):
            return .copy(from: v1, to: v2, count: new)
        default:
            return self
        }
    }
}
