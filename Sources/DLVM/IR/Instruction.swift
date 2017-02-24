//
//  Instruction.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

public struct Use {

    public enum Kind {
        case argument(Def<Argument>)
        case local(Def<Operation>)
        case global(Def<GlobalValue>)
        case literal(LiteralValue)
    }

    public var shape: TensorShape
    public var type: DataType
    public var kind: Kind

    public init(shape: TensorShape, type: DataType, kind: Kind) {
        self.shape = shape
        self.type = type
        self.kind = kind
    }

    public init(kind: Kind) {
        switch kind {
        case .local(let def):
            self.init(shape: def.shape, type: def.type, kind: kind)
        case .global(let def):
            self.init(shape: def.shape, type: def.type, kind: kind)
        case .argument(let def):
            self.init(shape: def.shape, type: def.type, kind: kind)
        case .literal(let lit):
            self.init(shape: lit.shape, type: lit.type, kind: kind)
        }
    }

}

// MARK: - Factory methods
public extension Use {

    static func local(_ definition: Def<Operation>) -> Use {
        return Use(kind: .local(definition))
    }

    static func global(_ definition: Def<GlobalValue>) -> Use {
        return Use(kind: .global(definition))
    }

    static func argument(_ definition: Def<Argument>) -> Use {
        return Use(kind: .argument(definition))
    }

    static func literal(_ literal: Literal, shape: TensorShape, type: DataType) -> Use {
        return Use(kind: .literal(LiteralValue(shape: shape, type: type, literal: literal)))
    }

    static func literal(_ literalValue: LiteralValue) -> Use {
        return Use(kind: .literal(literalValue))
    }

}

// MARK: - Value properties
public extension Use {

    var definition: AnyDef? {
        switch kind {
        case let .global(def): return def
        case let .local(def): return def 
        case let .argument(def): return def
        case .literal: return nil
        }
    }

    var value: Value {
        switch kind {
        case let .global(def): return def
        case let .local(def): return def
        case let .argument(def): return def
        case let .literal(lit): return lit
        }
    }

    var name: String? {
        switch kind {
        case .local(let def as Named),
             .global(let def as Named),
             .argument(let def as Named):
            return def.name
        case .literal:
            return nil
        }
    }
}

public class Instruction : IRObject {
    public enum Kind {
        case control(Control)
        case operation(Def<Operation>)
    }
    public typealias Parent = BasicBlock
    public weak internal(set) var parent: BasicBlock?
    public var kind: Kind

    public required init(kind: Kind) {
        self.kind = kind
    }

    public static func control(_ control: Control) -> Instruction {
        return self.init(kind: .control(control))
    }

    public static func operation(_ operation: Def<Operation>) -> Instruction {
        return self.init(kind: .operation(operation))
    }
}

public enum Control {
    /// Store use to global value
    case store(Use, to: Def<GlobalValue>)
    /// Yield value to output
    case yield(Use, to: Def<Output>)
    /// Unconditionally branch to basic block
    case br(BasicBlock)
    /// Conditional branch depending on the value
    case condBr(Use, BasicBlock, BasicBlock)
    /// Return
    case ret(Use?)
}

public enum Operation {
    /// Pull a value from the recurrent input batch in the current epoch
    /// - Precondition: placeholder must be recurrent
    case pull(Def<Placeholder>, BasicBlock, BasicBlock)
    /// Get the value of the non-recurrent input
    /// - Precondition: placeholder must **not** be recurrent
    case get(Def<Placeholder>)
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
    case matMul(Use, Use)
    /// Concatenation operation
    case concat(TensorShape, DataType, [Use], axis: Int)
    /// Phi node of SSA form
    case phi(TensorShape, DataType, [(Use, BasicBlock)])
    /// Type cast operation
    case typeCast(Use, DataType)
    /// Shape cast operation
    case shapeCast(Use, TensorShape)
    /// Subtensor addressing
    case subtensor(Use, TensorIndex)
    /// Intrinsic
    case intrinsic(TensorShape, DataType, Intrinsic, [Use])
    /// Element in the immediate dimension
    case element(Use, Int)
    /// Function call
    case call(TensorShape, DataType, Function, [Use])
    /// Differentiate
    case diff(TensorShape, DataType, Function, Use, wrt: Int)
}

extension Instruction {
    public var name: String? {
        guard case let .operation(def) = kind else { return nil }
        return def.name
    }
}

extension Use : Equatable {
    public static func ==(lhs: Use, rhs: Use) -> Bool {
        return lhs.definition === rhs.definition
            && lhs.name == rhs.name
            && lhs.shape == rhs.shape
            && lhs.type == rhs.type
    }
}

public extension Instruction {
    var isTerminator: Bool {
        switch kind {
        case .control(let ctrl): return ctrl.isTerminator
        case .operation(let def): return def.value.isTerminator
        }
    }
}

public extension Control {
    var isTerminator: Bool {
        switch self {
        case .br, .condBr, .ret:
            return true
        default:
            return false
        }
    }
}

public extension Operation {
    var isTerminator: Bool {
        switch self {
        case .pull:
            return true
        default:
            return false
        }
    }
}

extension Operation : Value {

    public var type: DataType {
        switch self {
        case let .binary(.associative(.arithmetic), op1, _):
            return op1.type
        case .binary(.associative(.boolean), _, _),
             .binary(.comparison, _, _):
            return .bool
        case let .matMul(op1, _):
            return op1.type
        case let .unary(_, op),
             let .reduce(_, op, _),
             let .scan(_, op, _):
            return op.type
        case let .phi(_, type, _):
            return type
        case let .concat(_, type, _, _):
            return type
        case let .typeCast(_, t):
            return t
        case let .shapeCast(op, _):
            return op.type
        case let .pull(ph, _, _):
            return ph.type
        case let .get(ph):
            return ph.type
        case let .call(_, type, _, _),
             let .diff(_, type, _, _, _):
            return type
        case let .element(op, _),
             let .subtensor(op, _):
            return op.type
        case let .intrinsic(_, type, _, _):
            return type
        }
    }

    public var shape: TensorShape {
        switch self {
        case let .matMul(op1, op2):
            return op1.shape.matrixMultiplied(with: op2.shape) ?? op1.shape
        case let .binary(_, op1, op2):
            return op1.shape.mutuallyBroadcasted(with: op2.shape) ?? op1.shape
        case let .unary(_, op),
             let .scan(_, op, axis: _):
            return op.shape
        case .reduce(_, _, axis: nil):
            return .scalar
        case let .reduce(_, op, axis: axis?):
            return op.shape.droppingDimension(axis)
        case let .phi(shape, _, _):
            return shape
        case let .concat(shape, _, _, _):
            return shape
        case let .typeCast(op, _):
            return op.shape
        case let .shapeCast(_, s):
            return s
        case let .pull(ph, _, _):
            return ph.shape
        case let .get(ph):
            return ph.shape
        case let .call(shape, _, _, _),
             let .diff(shape, _, _, _, _):
            return shape
        case let .element(op, _):
            return op.shape.dropFirst()
        case let .subtensor(op, idx):
            return op.shape[idx] ?? op.shape
        case let .intrinsic(shape, _, _, _):
            return shape
        }
    }

    public static var scope: Scope = .local

}

extension Control : User {
    public var operands: [Use] {
        switch self {
        case .condBr(let op, _, _),
             .yield(let op, _),
             .store(let op, _),
             .ret(let op?):
            return [op]
        default:
            return []
        }
    }
}

extension Operation : User {
    public var operands: [Use] {
        switch self {
        case let .binary(_, op1, op2),
             let .matMul(op1, op2):
            return [op1, op2]
        case .concat(_, _, let uses, axis: _):
            return uses
        case let .unary(_, op),
             let .reduce(_, op, _),
             let .scan(_, op, _),
             let .shapeCast(op, _),
             let .typeCast(op, _):
            return [op]
        case let .phi(_, _, incomings):
            return incomings.map{$0.0}
        case .call(_, _, _, let ops),
             .intrinsic(_, _, _, let ops):
            return ops
        case .pull, .get, .diff:
            return []
        case let .subtensor(op, _),
             let .element(op, _):
            return [op]
        }
    }
}

extension Instruction : User {
    public var operands: [Use] {
        switch kind {
        case .control(let ctrl): return ctrl.operands
        case .operation(let oper): return oper.operands
        }
    }
}

public extension Def where ValueType : User {
    public var operands: [Use] {
        return value.operands
    }
}

public extension Instruction {
    func substituting(_ actualUse: Use, for use: Use) -> Instruction {
        switch kind {
        case let .control(ctrl):
            return .control(ctrl.substituting(actualUse, for: use))
        case let .operation(def):
            let oper = def.value.substituting(actualUse, for: use)
            let newDef = Def<Operation>(name: def.name, value: oper)
            return .operation(newDef)
        }
    }
}

public extension Control {
    func substituting(_ actualUse: Use, for use: Use) -> Control {
        switch self {
        case .store(use, to: let dest):
            return .store(actualUse, to: dest)
        case .condBr(use, let thenBB, let elseBB):
            return .condBr(actualUse, thenBB, elseBB)
        case .yield(use, to: let dest):
            return .yield(actualUse, to: dest)
        case .ret(use?):
            return .ret(actualUse)
        default:
            return self
        }
    }
}

public extension Operation {
    func substituting(_ actualUse: Use, for use: Use) -> Operation {
        let condSubst = {$0 == use ? actualUse : $0}
        switch self {
        case .unary(let fun, use):
            return .unary(fun, actualUse)
        case .binary(let fun, use, let use2):
            return .binary(fun, actualUse, use2)
        case .binary(let fun, let use1, use):
            return .binary(fun, use1, actualUse)
        case .binary(let fun, use, use):
            return .binary(fun, actualUse, actualUse)
        case let .concat(shape, type, uses, axis: axis):
            return .concat(shape, type, uses.map(condSubst), axis: axis)
        case let .phi(shape, type, uses):
            return .phi(shape, type, uses.map{(condSubst($0), $1)})
        case .reduce(let fun, use, axis: let axis):
            return .reduce(fun, actualUse, axis: axis)
        case .matMul(use, let use2):
            return .matMul(actualUse, use2)
        case .matMul(let use1, use):
            return .matMul(use1, actualUse)
        case .matMul(use, use):
            return .matMul(actualUse, actualUse)
        case .shapeCast(use, let shape):
            return .shapeCast(actualUse, shape)
        case .typeCast(use, let type):
            return .typeCast(actualUse, type)
        default:
            return self
        }
    }
}
