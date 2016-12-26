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
        case lt, leq, gt, geq, eq
    }
    public enum BinaryOperator {
        case add, sub, mul, min, max
    }
    public enum ActivationFunction {
        case sigmoid, relu, tanh
    }
    public enum TransformationFunction {
        case log, softmax
    }
    public enum Kind {
        case negate(Operand)
        case binaryOp(BinaryOperator, Operand, Operand)
        case compare(ComparisonOperator, Operand, Operand)
        case dotProduct(TensorVariable, TensorVariable)
        case random(ScalarVariable, ScalarVariable)
        case product(TensorVariable, TensorVariable)
        case activation(ActivationFunction, TensorVariable)
        case transformation(TransformationFunction, TensorVariable)
        case concat([TensorVariable])
        case phi([Variable])
        case condBranch(Operand, then: BasicBlock, else: BasicBlock)
        case uncondBranch(BasicBlock)
        case output(TensorVariable)
    }
    public let kind: Kind
    public weak var parent: BasicBlock? = nil

    /// Initialize a standalone instruction by specifying its kind
    ///
    /// - Parameter kind: kind of instruction
    public init(kind: Kind) {
        self.kind = kind
    }
}

// MARK: - VariableProducer
extension Instruction : VariableProducer {

    public func makeVariable(named name: String) -> Variable {
        switch kind {
        /// Immediate-only instructions yield scalars
        case let .negate(op as Immediate),
             let .binaryOp(_, op as Immediate, _ as Immediate):
            let type: ScalarType
            switch op {
            case .bool:  type = .bool
            case .int:   type = .int
            case .float: type = .float
            }
            return ScalarVariable(name: name, type: type, definition: self)

        /// Scalar-only instructions
        case let .negate(op as ScalarVariable),
             let .binaryOp(_, op as ScalarVariable, _ as ScalarVariable):
            return ScalarVariable(name: name, type: op.type, definition: self)

        /// Tensor instructions
        case let .negate(op as TensorVariable),
             let .binaryOp(_, op as TensorVariable, _),
             let .binaryOp(_, _, op as TensorVariable),
             let .dotProduct(op, _),
             let .activation(_, op),
             let .transformation(_, op):
            return TensorVariable(name: name, dataType: op.dataType,
                          shape: op.shape, definition: self)

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
                return UnavailableVariable.shared
            }
            
        default: /// TODO
            return UnavailableVariable.shared
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
