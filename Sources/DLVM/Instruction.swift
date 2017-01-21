//
//  Instruction.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

public protocol LexicallyConvertible {
    static var lexicon: [String : Self] { get }
}

public enum LogicalPredicate {
    case and, or, xor
}

public enum ComparisonPredicate {
    case lessThan, lessThanOrEqualTo
    case greaterThan, greaterThanOrEqualTo
    case equalTo, notEqualTo
}

public enum ArithmeticOperator {
    case add, subtract, multiply, divide, min, max
    case truncateDivide, floorDivide, mod, power, mean
}

public enum ElementwiseFunction {
    case sigmoid, relu, tanh
    case log, exp, neg, sign, square, sqrt, round, rsqrt, ceil, floor
    case tan, cos, sin, acos, asin, atan
    case lgamma, digamma, erf, erfc, rint

}

public enum BinaryReductionFunction {
    case crossEntropy
}

public enum ReductionFunction {
    case logical(LogicalPredicate)
    case comparison(ComparisonPredicate)
    case arithmetic(ArithmeticOperator)
}

public enum AggregateFunction {
    case softmax, logSoftmax
    case scan(ReductionFunction)
}

public class Instruction : IRObject {
    public weak var parent: BasicBlock?
    fileprivate init() {}
}

/// Instruction base class
public class DefiningInstruction : Instruction, NamedValue {
    public var name: String
    public var type: DataType
    public var shape: TensorShape

    fileprivate init(name: String, type: DataType, shape: TensorShape) {
        self.name = name
        self.type = type
        self.shape = shape
    }
}

public class UnaryInstruction : DefiningInstruction {
    public var operand: Value

    fileprivate init(name: String, type: DataType, shape: TensorShape, operand: Value) {
        self.operand = operand
        super.init(name: name, type: type, shape: shape)
    }
}

public class BinaryInstruction : DefiningInstruction {
    public var firstOperand, secondOperand: Value

    fileprivate init(name: String, type: DataType, shape: TensorShape,
                     firstOperand: Value, secondOperand: Value) {
        self.firstOperand = firstOperand
        self.secondOperand = secondOperand
        super.init(name: name, type: type, shape: shape)
    }
}

public class UnaryCallInstruction<Function> : UnaryInstruction {
    public var function: Function

    public init(name: String, type: DataType, shape: TensorShape,
                function: Function, operand: Value) {
        self.function = function
        super.init(name: name, type: type, shape: shape, operand: operand)
    }
}

public class BinaryCallInstruction<Function> : BinaryInstruction {
    public var function: Function

    public init(name: String, type: DataType, shape: TensorShape,
                function: Function, firstOperand: Value, secondOperand: Value) {
        self.function = function
        super.init(name: name, type: type, shape: shape,
                   firstOperand: firstOperand, secondOperand: secondOperand)
    }
}

public class ReductionInstruction : UnaryCallInstruction<ReductionFunction> {
    public init(name: String, function: ReductionFunction, operand: Value) {
        super.init(name: name, type: operand.type, shape: .scalar,
                   function: function, operand: operand)
    }
}

public class HomomorphicUnaryInstruction<Function> : UnaryCallInstruction<Function> {
    public init(name: String, function: Function, operand: Value) {
        super.init(name: name, type: operand.type, shape: operand.shape,
                   function: function, operand: operand)
    }
}

public class HomomorphicBinaryInstruction<Function> : BinaryCallInstruction<Function> {
    public init(name: String, function: Function, firstOperand: Value, secondOperand: Value) {
        super.init(name: name, type: firstOperand.type, shape: firstOperand.shape,
                   function: function, firstOperand: firstOperand, secondOperand: secondOperand)
    }
}

public typealias ElementwiseTransformationInstruction = HomomorphicUnaryInstruction<ElementwiseFunction>
public typealias BinaryReductionInstruction = HomomorphicBinaryInstruction<BinaryReductionFunction>
public typealias ArithmeticInstruction = HomomorphicBinaryInstruction<ArithmeticOperator>

public class ComparisonInstruction : BinaryInstruction {
    public var predicate: ComparisonPredicate

    public init(name: String, predicate: ComparisonPredicate,
                firstOperand: Value, secondOperand: Value) {
        self.predicate = predicate
        var newType = firstOperand.type
        newType.base = .bool
        newType.size = 1
        super.init(name: name, type: newType, shape: firstOperand.shape,
                   firstOperand: firstOperand, secondOperand: secondOperand)
    }
}

public final class AggregateTransformationInstruction : HomomorphicUnaryInstruction<AggregateFunction> {
    public override init(name: String, function: AggregateFunction, operand: Value) {
        super.init(name: name, function: function, operand: operand)
        /// For scanning with logical and comparison operators,
        /// the result type is boolean
        switch function {
        case .scan(.logical(_)), .scan(.comparison(_)): type = .bool
        default: break
        }
    }
}

public final class TensorMultiplicationInstruction : BinaryInstruction {
    public init(name: String, firstOperand: Value, secondOperand: Value) {
        let newShape = (firstOperand.shape ⊗ secondOperand.shape) ?? firstOperand.shape
        super.init(name: name, type: firstOperand.type, shape: newShape,
                   firstOperand: firstOperand, secondOperand: secondOperand)
    }
}

public final class MatrixMultiplicationInstruction : BinaryInstruction {
    public init(name: String, firstOperand: Value, secondOperand: Value) {
        let newShape = firstOperand.shape.matrixMultiplied(by: secondOperand.shape) ?? firstOperand.shape
        super.init(name: name, type: firstOperand.type, shape: newShape,
                   firstOperand: firstOperand, secondOperand: secondOperand)
    }
}

public final class ConcatenationInstruction : DefiningInstruction {
    public var operands: [Value]
    public var axis: Int

    public init(name: String, operands: [Value], axis: Int) {
        precondition(!operands.isEmpty)
        self.operands = Array(operands)
        self.axis = axis
        let firstShape = operands[0].shape
        let newShape = operands.dropFirst().reduce(firstShape, { acc, x in
            acc?.concatenating(with: x.shape, alongDimension: axis)
        }) ?? firstShape
        super.init(name: name, type: operands[0].type, shape: newShape)
    }
}

public final class ShapeCastInstruction : DefiningInstruction {
    public var operand: Value
    public var targetShape: TensorShape

    public init(name: String, operand: Value, targetShape: TensorShape) {
        self.operand = operand
        self.targetShape = targetShape
        super.init(name: name, type: operand.type, shape: targetShape)
    }
}

public final class TypeCastInstruction : DefiningInstruction {
    public var operand: Value
    public var targetType: DataType

    public init(name: String, operand: Value, targetType: DataType) {
        self.operand = operand
        self.targetType = targetType
        super.init(name: name, type: targetType, shape: operand.shape)
    }
}

public final class LoadInstruction : DefiningInstruction {
    public var source: Value

    public init(name: String, source: Value) {
        self.source = source
        super.init(name: name, type: source.type, shape: source.shape)
    }
}

public final class StoreInstruction : Instruction {
    public var source: Value
    public var destination: GlobalValue

    public init(source: Value, destination: GlobalValue) {
        self.source = source
        self.destination = destination
    }
}
