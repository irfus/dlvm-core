//
//  Instruction.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

import DLVMTensor

public enum InstructionKind {
    /** Control flow **/
    /// Unconditionally branch to basic block
    case branch(BasicBlock, [Use])
    /// Conditional branch depending on the value
    case conditional(Use, BasicBlock, [Use], BasicBlock, [Use])
    /// Return
    case `return`(Use?)

    /** Tensor integration **/
    /// Scan operation with optional axis
    /// If axis is not given, scan is performed on contiguous elements
    case scan(AssociativeOp, Use, axis: Int?)
    /// Reduction operation with optional axis
    /// If axis is not given, reduction is performed on contiguous elements
    case reduce(AssociativeOp, Use, axis: Int?)
    /// Matrix multiplication operation
    case matrixMultiply(Use, Use)

    /** Tensor monomorphic transfer **/
    /// Monomorphic unary operation
    case unary(UnaryOp, Use)
    /// Monomorphic binary operation
    case binary(BinaryOp, Use, Use)

    /** Tensor manipulation **/
    /// Concatenation operation
    case concatenate([Use], axis: Int)
    /// Transpose
    case transpose(Use)

    /** Cast **/
    /// Type cast operation
    case dataTypeCast(Use, DataType)
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
    /// Function call
    case call(Use, [Use])
    /// Gradient call, using implicit automatic differentiation
    /// Will be replaced by `call` through GradientExpansion transform pass
    case gradient(Use, [Use])

    /** Memory **/
    /// Allocate stack memory, returning a pointer
    case allocate(Type, Use)
    /// Load value from pointer
    case load(Use)
    /// Store value to pointer
    case store(Use, to: Use)
    /// GEP
    case elementPointer(Use, [Use])
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

// MARK: - Control flow predicates
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
            guard case let .tensor(s1, t1) = v1.type.unaliased,
                  case let .tensor(s2, t2) = v2.type.unaliased,
                let shape = s1 <> s2, t1 == t2
                else { return .invalid }
            return .tensor(shape, t1)

        case let .binary(.associative(.boolean), v1, v2),
             let .binary(.comparison, v1, v2):
            guard case let .tensor(s1, t1) = v1.type.unaliased,
                  case let .tensor(s2, t2) = v2.type.unaliased,
                  let shape = s1 <> s2, t1 == t2
                else { return .invalid }
            return .tensor(shape, .bool)
            
        case let .matrixMultiply(v1, v2):
            guard case let .tensor(s1, t1) = v1.type.unaliased,
                  case let .tensor(s2, t2) = v2.type.unaliased,
                  let newShape = s1.matrixMultiplied(with: s2),
                  t1 == t2 else { return .invalid }
            return .tensor(newShape, t1)

        case let .unary(_, v1):
            return v1.type.isTensor ? v1.type : .invalid

        case let .reduce(_, v1, nil):
            guard case let .tensor(s1, t1) = v1.type.unaliased else { return .invalid }
            return .tensor(s1.dropFirst(), t1)

        case let .reduce(_, v1, axis: axis?):
            guard case let .tensor(s1, t1) = v1.type.unaliased,
                  axis < s1.rank
                else { return .invalid }
            return .tensor(s1.droppingDimension(axis), t1)

        case let .scan(_, v1, _):
            return v1.type.unaliased.isTensor ? v1.type : .invalid

        case let .concatenate(vv, axis):
            guard let first = vv.first,
                  case let .tensor(s1, t1) = first.type.unaliased
                else { return .invalid }
            /// Check simple, data type equality, and concatenability
            var accShape: TensorShape = s1
            for v in vv.dropFirst() {
                guard case let .tensor(shape, type) = v.type.unaliased,
                      type == t1,
                      let newShape = accShape.concatenating(with: shape,
                                                            alongDimension: axis)
                    else { return .invalid }
                accShape = newShape
            }
            return .tensor(accShape, t1)

        case let .transpose(v1):
            guard case let .tensor(s1, t1) = v1.type.unaliased else { return .invalid }
            return .tensor(s1.transpose, t1)
            
        case let .dataTypeCast(v1, dt):
            guard case let .tensor(s1, t1) = v1.type.unaliased,
                  t1.canCast(to: dt)
                else { return .invalid }
            return .tensor(s1, dt)
            
        case let .shapeCast(v1, s):
            guard case let .tensor(s1, t1) = v1.type.unaliased,
                  s1.contiguousSize == s.contiguousSize
                else { return .invalid }
            return .tensor(s, t1)

        case let .call(f, vv):
            let actual = vv.map{$0.type}
            switch f.type.unaliased {
            case let .function(actual, ret), let .pointer(.function(actual, ret)):
                return ret
            default:
                return .invalid
            }

        case let .gradient(f, vv):
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

        case let .allocate(type, _):
            return .pointer(type)

        case let .load(v):
            guard case let .pointer(t) = v.type.unaliased else { return .invalid }
            return t

        case let .elementPointer(v, ii):
            func gepType<C: Collection>(type: Type, indices: C) -> Type where C.Iterator.Element == Use {
                if indices.isEmpty { return type }
                if case let .pointer(t) = type.unaliased {
                    return gepType(type: t, indices: ii.dropFirst())
                } else {
                    return .invalid
                }
            }
            return gepType(type: v.type, indices: ii)

        case let .bitCast(_, t):
            return t

        case .store, .branch, .conditional, .return:
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
        case let .unary(_, op),
             let .reduce(_, op, _),
             let .scan(_, op, _),
             let .shapeCast(op, _),
             let .dataTypeCast(op, _),
             let .return(op?),
             let .extract(from: op, at: _),
             let .transpose(op),
             let .store(op, _),
             let .load(op),
             let .elementPointer(op, _),
             let .bitCast(op, _):
            return [op]
        case .concatenate(let ops, _),
             .call(_, let ops),
             .gradient(_, let ops),
             .branch(_, let ops):
            return ops
        case let .conditional(cond, _, thenArgs, _, elseArgs):
            return [cond] + thenArgs + elseArgs
        case .return(nil), .allocate:
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
    func substituting(_ newUse: Use, for old: Use) -> InstructionKind {
        let condSubst = {$0 == old ? newUse : $0}
        switch self {
        case .branch(let dest, let args):
            return .branch(dest, args.map(condSubst))
        case let .conditional(cond, thenBB, thenArgs, elseBB, elseArgs):
            let newCond = cond == old ? newUse : cond
            return .conditional(newCond,
                                thenBB, thenArgs.map(condSubst),
                                elseBB, elseArgs.map(condSubst))
        case .return(old?):
            return .return(newUse)
        case .unary(let fun, old):
            return .unary(fun, newUse)
        case .binary(let fun, old, old):
            return .binary(fun, newUse, newUse)
        case .binary(let fun, old, let use2):
            return .binary(fun, newUse, use2)
        case .binary(let fun, let use1, old):
            return .binary(fun, use1, newUse)
        case let .concatenate(uses, axis: axis):
            return .concatenate(uses.map(condSubst), axis: axis)
        case .transpose(old):
            return .transpose(newUse)
        case .reduce(let fun, old, axis: let axis):
            return .reduce(fun, newUse, axis: axis)
        case .matrixMultiply(old, let use2):
            return .matrixMultiply(newUse, use2)
        case .matrixMultiply(let use1, old):
            return .matrixMultiply(use1, newUse)
        case .matrixMultiply(old, old):
            return .matrixMultiply(newUse, newUse)
        case .shapeCast(old, let shape):
            return .shapeCast(newUse, shape)
        case .dataTypeCast(old, let type):
            return .dataTypeCast(newUse, type)
        case let .call(f, uses):
            return .call(f, uses.map(condSubst))
        case let .gradient(f, uses):
            return .gradient(f, uses.map(condSubst))
        case .extract(from: old, at: let i):
            return .extract(from: newUse, at: i)
        case .insert(old, to: old, at: let indices):
            return .insert(newUse, to: newUse, at: indices)
        case .insert(old, to: let use1, at: let indices):
            return .insert(newUse, to: use1, at: indices)
        case .insert(let use1, to: old, at: let indices):
            return .insert(use1, to: newUse, at: indices)
        case .bitCast(old, let targetT):
            return .bitCast(newUse, targetT)
        case .elementPointer(old, let indices):
            return .elementPointer(newUse, indices)
        case .store(old, to: let dest):
            return .store(newUse, to: dest)
        case .store(let val, to: old):
            return .store(val, to: newUse)
        case .load(old):
            return .load(newUse)
        default:
            return self
        }
    }
}
