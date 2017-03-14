//
//  Instruction.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

public enum InstructionKind {
    /// Store use to global value
    case store(Use, to: GlobalValue)
    /// Unconditionally branch to basic block
    case branch(BasicBlock, [Use])
    /// Conditional branch depending on the value
    case conditional(Use, BasicBlock, BasicBlock)
    /// Return
    case `return`(Use?)
    /// Scan operation with optional axis
    /// If axis is not given, scan is performed on contiguous elements
    case scan(AssociativeOp, Use, axis: Int?)
    /// Reduction operation with optional axis
    /// If axis is not given, reduction is performed on contiguous elements
    case reduce(AssociativeOp, Use, axis: Int?)
    /// Monomorphic unary operation
    case unary(UnaryOp, Use)
    /// Monomorphic binary operation
    case binary(BinaryOp, Use, Use)
    /// Matrix multiplication operation
    case matrixMultiply(Use, Use)
    /// Concatenation operation
    case concatenate([Use], axis: Int)
    /// Transpose
    case transpose(Use)
    /// Type cast operation
    case dataTypeCast(Use, DataType)
    /// Shape cast operation
    case shapeCast(Use, TensorShape)
    /// Subtensor addressing
    case subtensor(Use, TensorIndex)
    /// Element in the immediate dimension
    case tupleElement(Use, Int)
    /// Create tuple
    case tuple([Use])
    /// Function call
    case call(Function, [Use])
    /// Gradient call
    case gradient(Function, [Use])
}

public final class Instruction : IRSubUnit, Value {
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

extension InstructionKind : Value {

    public var type: Type {
        switch self {
        case let .binary(.associative(.arithmetic), v1, v2):
            guard case let .tensor(s1, t1) = v1.type,
                  case let .tensor(s2, t2) = v2.type,
                let shape = s1 <> s2, t1 == t2
                else { return .invalid }
            return .tensor(shape, t1)

        case let .binary(.associative(.boolean), v1, v2),
             let .binary(.comparison, v1, v2):
            guard case let .tensor(s1, t1) = v1.type,
                  case let .tensor(s2, t2) = v2.type,
                  let shape = s1 <> s2, t1 == t2
                else { return .invalid }
            return .tensor(shape, .bool)
            
        case let .matrixMultiply(v1, v2):
            guard case let .tensor(s1, t1) = v1.type,
                  case let .tensor(s2, t2) = v2.type,
                  let newShape = s1.matrixMultiplied(with: s2),
                  t1 == t2 else { return .invalid }
            return .tensor(newShape, t1)

        case let .unary(_, v1):
            return v1.type.isTensor ? v1.type : .invalid

        case let .reduce(_, v1, nil):
            guard case let .tensor(s1, t1) = v1.type else { return .invalid }
            return .tensor(s1.dropFirst(), t1)

        case let .reduce(_, v1, axis: axis?):
            guard case let .tensor(s1, t1) = v1.type,
                  axis < s1.rank
                else { return .invalid }
            return .tensor(s1.droppingDimension(axis), t1)

        case let .scan(_, v1, _):
            return v1.type.isTensor ? v1.type : .invalid

        case let .concatenate(vv, axis):
            guard let first = vv.first,
                  case let .tensor(s1, t1) = first.type
                else { return .invalid }
            /// Check simple, data type equality, and concatenability
            var accShape: TensorShape = s1
            for v in vv.dropFirst() {
                guard case let .tensor(shape, type) = v.type, type == t1,
                      let newShape = accShape.concatenating(with: shape,
                                                            alongDimension: axis)
                    else { return .invalid }
                accShape = newShape
            }
            return .tensor(accShape, t1)

        case let .transpose(v1):
            guard case let .tensor(s1, t1) = v1.type else { return .invalid }
            return .tensor(s1.transpose, t1)
            
        case let .dataTypeCast(v1, dt):
            guard case let .tensor(s1, t1) = v1.type,
                  t1.canCast(to: dt)
                else { return .invalid }
            return .tensor(s1, dt)
            
        case let .shapeCast(v1, s):
            guard case let .tensor(s1, t1) = v1.type,
                  s1.contiguousSize == s.contiguousSize
                else { return .invalid }
            return .tensor(s, t1)

        case let .call(f, vv):
            guard f.acceptsArguments(vv.map{$0.type})
                else { return .invalid }
            return f.result

        case let .gradient(f, vv):
            guard f.isDifferentiable, f.acceptsArguments(vv.map{$0.type})
                else { return .invalid }
            if f.arguments.isEmpty { return .void }
            if f.arguments.count == 1 { return f.arguments[0].type }
            return .tuple(vv.map{$0.type})

        case let .tuple(vv):
            return .tuple(vv.map{$0.type})
            
        case let .tupleElement(v, i):
            guard case let .tuple(subtypes) = v.type,
                  subtypes.indices.contains(i)
                else { return .invalid }
            return subtypes[i]

        case let .subtensor(v, index):
            guard case let .tensor(s1, t1) = v.type,
                  index.count < s1.rank
                else { return .invalid }
            return .tensor(s1.dropFirst(index.count), t1)

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
             let .matrixMultiply(op1, op2):
            return [op1, op2]
        case let .unary(_, op),
             let .reduce(_, op, _),
             let .scan(_, op, _),
             let .shapeCast(op, _),
             let .dataTypeCast(op, _),
             let .conditional(op, _, _),
             let .store(op, _),
             let .return(op?),
             let .subtensor(op, _),
             let .tupleElement(op, _),
             let .transpose(op):
            return [op]
        case .concatenate(let ops, _),
             .call(_, let ops),
             .gradient(_, let ops),
             .tuple(let ops),
             .branch(_, let ops):
            return ops
        case .return(nil):
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
        case .store(old, to: let dest):
            return .store(newUse, to: dest)
        case .conditional(old, let thenBB, let elseBB):
            return .conditional(newUse, thenBB, elseBB)
        case .return(old?):
            return .return(newUse)
        case .unary(let fun, old):
            return .unary(fun, newUse)
        case .binary(let fun, old, let use2):
            return .binary(fun, newUse, use2)
        case .binary(let fun, let use1, old):
            return .binary(fun, use1, newUse)
        case .binary(let fun, old, old):
            return .binary(fun, newUse, newUse)
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
        case let .tuple(uses):
            return .tuple(uses.map(condSubst))
        case .tupleElement(old, let i):
            return .tupleElement(newUse, i)
        case .subtensor(old, let index):
            return .subtensor(newUse, index)
        default:
            return self
        }
    }
}
