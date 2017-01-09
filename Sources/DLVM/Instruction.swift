//
//  Instruction.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

open class Instruction : IRObject {
    public typealias Parent = BasicBlock
    public enum ComparisonOperator {
        case lt, leq, gt, geq, eq, neq
    }
    public enum ArithmeticOperator {
        case add, sub, mul, div, min, max
    }
    public enum ActivationFunction {
        case sigmoid, relu, tanh
    }
    public enum TransformationFunction {
        case log, softmax
    }
    public enum Kind {
        case negate(Operand)
        case arithOp(ArithmeticOperator, Operand, Operand)
        case compare(ComparisonOperator, Operand, Operand)
        case dotProduct(TensorVariable, TensorVariable)
        case product(TensorVariable, TensorVariable)
        case activation(ActivationFunction, TensorVariable)
        case transformation(TransformationFunction, TensorVariable)
        case concat([TensorVariable], dimension: Int)
        case phi([VariableOperand])
        case shapeCast(TensorShape, TensorVariable)
        case condBranch(Operand, then: BasicBlock, else: BasicBlock)
        case uncondBranch(BasicBlock)
        case output(TensorVariable)
    }
    public let kind: Kind
    public internal(set) weak var variable: VariableOperand?
    public internal(set) weak var parent: BasicBlock?

    /// Initialize a standalone instruction by specifying its kind
    ///
    /// - Parameter kind: kind of instruction
    public init(kind: Kind) {
        self.kind = kind
    }
}

// MARK: - VariableProducer
extension Instruction : VariableProducer {

    public func makeVariable(named name: String) -> VariableOperand? {
        switch kind {
        /// Immediate-only instructions yield scalars
        case let .negate(op as ImmediateOperand),
             let .arithOp(_, op as ImmediateOperand, _ as ImmediateOperand):
            let type: ScalarType
            switch op {
            case .bool:  type = .bool
            case .int:   type = .int
            case .float: type = .float
            }
            return ScalarVariable(name: name, type: type, definition: self)

        /// Scalar-only instructions
        case let .negate(op as ScalarVariable),
             let .arithOp(_, op as ScalarVariable, _ as ScalarVariable):
            return ScalarVariable(name: name, type: op.type, definition: self)

        /// Tensor instructions
        case let .negate(op as TensorVariable),
             let .arithOp(_, op as TensorVariable, _),
             let .arithOp(_, _, op as TensorVariable),
             let .dotProduct(op, _),
             let .activation(_, op),
             let .transformation(_, op):
            return TensorVariable(name: name, dataType: op.dataType,
                                  shape: op.shape, definition: self)

        case let .product(lhs, rhs):
            let newShape = lhs.shape.product(with: rhs.shape)!
            return TensorVariable(name: name, dataType: lhs.dataType,
                                  shape: newShape, definition: self)

        /// Cast instruction
        case let .shapeCast(targetShape, variable):
            return TensorVariable(name: variable.name, dataType: variable.dataType,
                                  shape: targetShape, definition: self)

        /// Phi node
        /// Args are either all tensors (of the same shape) or all scalars
        case let .phi(vars):
            let firstVar = vars[0]
            switch firstVar {
            case let arg as TensorVariable:
                return TensorVariable(name: name, dataType: arg.dataType,
                                      shape: arg.shape, definition: self)
            case let arg as ScalarVariable:
                return ScalarVariable(name: name, type: arg.type, definition: self)
            default:
                preconditionFailure("Unsupported variable type")
            }

        case let .concat(variables, dimension: dim):
            let firstShape = variables[0].shape
            let newShape = variables.dropFirst().reduce(firstShape) { acc, x in
                acc.concatenating(with: x.shape, alongDimension: dim)!
            }
            return TensorVariable(name: name, dataType: variables[0].dataType,
                                  shape: newShape, definition: self)

        default: /// TODO: many cases like `compare` are not handled!
            return nil
        }
    }

}

// MARK: - Hashable
extension Instruction : Hashable {

    /// Equatable by reference
    public static func == (lhs: Instruction, rhs: Instruction) -> Bool {
        return lhs === rhs
    }
    
    /// Hashable by object identifier
    public var hashValue: Int {
        return ObjectIdentifier(self).hashValue
    }
    
}
