//
//  Instruction.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

public enum ComparisonPredicate {
    case lessThan, lessThanOrEqualTo
    case greaterThan, greaterThanOrEqualTo
    case equalTo, notEqualTo
}

public enum BooleanFunction {
    case and, or, xor
}

public enum ArithmeticOperator {
    case add, subtract, multiply, divide, min, max
    case truncateDivide, floorDivide, modulo, power, mean
}

public enum ElementwiseFunction {
    case sigmoid, tanh
    case log, exp, neg, sign, square, sqrt, round, rsqrt, ceil, floor
    case tan, cos, sin, acos, asin, atan
    case lgamma, digamma, erf, erfc, rint
}

public enum FunctionKind {
    case unary(UnaryFunction)   /// Monomorphic
    case binary(BinaryFunction) /// Monomorphic
    case matrixMultiplication   /// Matrix multiplication
    case reduction              /// Reducing 1 dimension
}

public enum IntegrationFunction {
    case softmax, logSoftmax, argmax, argmin
}

public enum UnaryFunction {
    case elementwise(ElementwiseFunction)
    case integration(IntegrationFunction)
    case scan(BinaryFunction)
}

public enum AssociativeFunction {
    case boolean(BooleanFunction)
    case arithmetic(ArithmeticOperator)
}

public enum BinaryFunction {
    case associative(AssociativeFunction)
    case comparison(ComparisonPredicate)
}

public class Def<ValueType : Value> : ManagedUsee, Named {
    public typealias UserType = Instruction
    public var name: String
    public var shape: TensorShape
    public var type: DataType
    public var value: ValueType
    public var users: NamedObjectSet<Instruction> = []
    public init(name: String, value: ValueType) {
        self.name = name
        self.shape = value.shape
        self.type = value.type
        self.value = value
    }
}

public struct Use {
    public enum Kind {
        case argument(Def<Argument>)
        case local(Def<Operation>)
        case global(Def<GlobalValue>)
        case placeholder(Def<Placeholder>)
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
        case .placeholder(let def):
            self.init(shape: def.shape, type: def.type, kind: kind)
        case .argument(let def):
            self.init(shape: def.shape, type: def.type, kind: kind)
        case .literal(let lit):
            self.init(shape: lit.shape, type: lit.type, kind: kind)
        }
    }

    public var definition: AnyObject? {
        switch kind {
        case let .global(def): return def
        case let .local(def): return def
        case let .placeholder(def): return def
        case let .argument(def): return def
        case .literal: return nil
        }
    }

    public var name: String? {
        switch kind {
        case .local(let def):
            return def.name
        case .global(let def):
            return def.name
        case .placeholder(let def):
            return def.name
        case .argument(let def):
            return def.name
        default:
            return nil
        }
    }
}

public enum Instruction {
    case control(Control)
    case operation(Def<Operation>)
}

public enum Control {
    case store(Use, to: GlobalValue)
    case export(Use)
    case br(BasicBlock)
    case condBr(Use, BasicBlock, BasicBlock)
    case dequeueBr(Def<Placeholder>, BasicBlock, BasicBlock)
    case ret
}

public enum Operation {
    case reduce(AssociativeFunction, Use, axis: Int?)
    case unary(IntegrationFunction, Use)
    case binary(BinaryFunction, Use, Use)
    case matrixMultiply(Use, Use)
    case concat([Use], axis: Int)
    case phi([(Use, BasicBlock)])
    case typeCast(Use, DataType)
    case shapeCast(Use, TensorShape)
}

extension Instruction {
    public var name: String? {
        guard case let .operation(def) = self else {
            return nil
        }
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

extension Operation : Value {

    public var type: DataType {
        switch self {
        case let .binary(.associative(.arithmetic), op1, _):
            return op1.type
        case .binary(.associative(.boolean), _, _),
             .binary(.comparison, _, _):
            return .bool
        case let .matrixMultiply(op1, _):
            return op1.type
        case let .unary(_, op),
             let .reduce(_, op, _):
            return op.type
        case let .phi(ops):
            return ops[0].0.type
        case let .concat(ops, _):
            return ops[0].type
        case let .typeCast(_, t):
            return t
        case let .shapeCast(op, _):
            return op.type
        }
    }

    public var shape: TensorShape {
        switch self {
        case let .binary(_, op1, _),
             let .matrixMultiply(op1, _):
            return op1.shape
        case let .unary(_, op),
             let .reduce(_, op, _):
            return op.shape
        case let .phi(ops):
            return ops[0].0.shape
        case let .concat(ops, _):
            return ops[0].shape
        case let .typeCast(op, _):
            return op.shape
        case let .shapeCast(_, s):
            return s
        }
    }

    public static var scope: Scope = .local

}

extension Control : User {
    public var operands: [Use] {
        switch self {
        case .condBr(let op, _, _),
             .export(let op),
             .store(let op, _):
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
             let .matrixMultiply(op1, op2):
            return [op1, op2]
        case .concat(let uses, axis: _):
            return uses
        case let .unary(_, op),
             let .reduce(_, op, _),
             let .shapeCast(op, _),
             let .typeCast(op, _):
            return [op]
        case let .phi(incomings):
            return incomings.map{$0.0}
        }
    }
}

extension Instruction : User {
    public var operands: [Use] {
        switch self {
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

extension Instruction {
    func substituting(_ actualUse: Use, for use: Use) -> Instruction {
        switch self {
        case let .control(ctrl):
            return .control(ctrl.substituting(actualUse, for: use))
        case let .operation(def):
            let oper = def.value.substituting(actualUse, for: use)
            let newDef = Def<Operation>(name: def.name, value: oper)
            return .operation(newDef)
        }
    }
}

extension Control {
    func substituting(_ actualUse: Use, for use: Use) -> Control {
        switch self {
        case .store(use, to: let dest):
            return .store(actualUse, to: dest)
        case .condBr(use, let thenBB, let elseBB):
            return .condBr(actualUse, thenBB, elseBB)
        case .export(use):
            return .export(actualUse)
        default:
            return self
        }
    }
}

extension Operation {
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
        case let .concat(uses, axis: axis):
            return .concat(uses.map(condSubst), axis: axis)
        case let .phi(uses):
            return .phi(uses.map{(condSubst($0), $1)})
        case .reduce(let fun, use, axis: let axis):
            return .reduce(fun, actualUse, axis: axis)
        case .matrixMultiply(use, let use2):
            return .matrixMultiply(actualUse, use2)
        case .matrixMultiply(let use1, use):
            return .matrixMultiply(use1, actualUse)
        case .matrixMultiply(use, use):
            return .matrixMultiply(actualUse, actualUse)
        case .shapeCast(use, let shape):
            return .shapeCast(actualUse, shape)
        case .typeCast(use, let type):
            return .typeCast(actualUse, type)
        default:
            return self
        }
    }
}
