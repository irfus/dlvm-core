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

public enum LogicOperator {
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

public enum BinaryIntegrationFunction {
    case crossEntropy
}

public enum ReductionFunction {
    case logical(LogicOperator)
    case arithmetic(ArithmeticOperator)
}

public enum AggregationFunction {
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

/// Abstract class for unary instruction
public class UnaryInstruction : DefiningInstruction {
    public var operand: Value

    fileprivate init(name: String, type: DataType, shape: TensorShape, operand: Value) {
        self.operand = operand
        super.init(name: name, type: type, shape: shape)
    }
}

/// Abstract class for binary instruction
public class BinaryInstruction : DefiningInstruction {
    public var firstOperand, secondOperand: Value

    fileprivate init(name: String, type: DataType, shape: TensorShape,
                     firstOperand: Value, secondOperand: Value) {
        self.firstOperand = firstOperand
        self.secondOperand = secondOperand
        super.init(name: name, type: type, shape: shape)
    }
}

/// Abstract class for unary function calls
public class UnaryCallInstruction<Function> : UnaryInstruction {
    public var function: Function

    fileprivate init(name: String, type: DataType, shape: TensorShape,
                function: Function, operand: Value) {
        self.function = function
        super.init(name: name, type: type, shape: shape, operand: operand)
    }
}

/// Abstract class for binary function calls
public class BinaryCallInstruction<Function> : BinaryInstruction {
    public var function: Function

    fileprivate init(name: String, type: DataType, shape: TensorShape,
                function: Function, firstOperand: Value, secondOperand: Value) {
        self.function = function
        super.init(name: name, type: type, shape: shape,
                   firstOperand: firstOperand, secondOperand: secondOperand)
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

/// Shape-preserving, type-preserving (homomorphic) transformation instructions
public typealias ElementwiseInstruction = HomomorphicUnaryInstruction<ElementwiseFunction>
public typealias AggregationInstruction = HomomorphicUnaryInstruction<AggregationFunction>
public typealias BinaryReductionInstruction = HomomorphicBinaryInstruction<BinaryIntegrationFunction>
public typealias ArithmeticInstruction = HomomorphicBinaryInstruction<ArithmeticOperator>
public typealias LogicInstruction = HomomorphicBinaryInstruction<LogicOperator>

/// Reduction instruction
/// - Note: We are assuming that reduction instruction takes any tensor and apply
/// the reduction function between elements contiguously in memory.
/// At the end of the day we will want shape-based reduction, e.g. reducing a
/// dimension from the shape. But this is not a priority.
/// Q: Why do we still want this feature? A: TensorFlow has it.
public class ReductionInstruction : UnaryCallInstruction<ReductionFunction> {
    public init(name: String, function: ReductionFunction, operand: Value) {
        super.init(name: name, type: operand.type, shape: .scalar,
                function: function, operand: operand)
    }
}

/// Comparison instruction
/// - shape: [A] -> [A]
/// - type: _ -> Bool
public final class ComparisonInstruction : BinaryCallInstruction<ComparisonPredicate> {
    public init(name: String, function: ComparisonPredicate,
                firstOperand: Value, secondOperand: Value) {
        super.init(name: name, type: .bool, shape: firstOperand.shape,
                   function: function, firstOperand: firstOperand, secondOperand: secondOperand)
    }
}

/// Generic tensor multiplication instruction (GETT operation)
public final class TensorMultiplicationInstruction : BinaryInstruction {
    public init(name: String, firstOperand: Value, secondOperand: Value) {
        let newShape = (firstOperand.shape ⊗ secondOperand.shape) ?? firstOperand.shape
        super.init(name: name, type: firstOperand.type, shape: newShape,
                   firstOperand: firstOperand, secondOperand: secondOperand)
    }
}

/// Matrix multiplication instruction (GEMM operation)
/// - Note: This only applies to the two inner dimensions
public final class MatrixMultiplicationInstruction : BinaryInstruction {
    public init(name: String, firstOperand: Value, secondOperand: Value) {
        let newShape = firstOperand.shape.matrixMultiplied(by: secondOperand.shape) ?? firstOperand.shape
        super.init(name: name, type: firstOperand.type, shape: newShape,
                   firstOperand: firstOperand, secondOperand: secondOperand)
    }
}

/// Concatenation instruction
/// Concatenates multiple tensors along an axis dimension
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

/// Shape-cast instruction
/// Casts the shape of a tensor to another
/// - Precondition: Target shape must be of the same contiguous size
/// - Note: We will want to be able to shape-cast between different contiguous
/// sizes (Q: Reason? A: TensorFlow does.). This should be done after we
/// introduce **slicing**-related instructions.
public final class ShapeCastInstruction : DefiningInstruction {
    public var operand: Value
    public var targetShape: TensorShape

    public init(name: String, operand: Value, targetShape: TensorShape) {
        self.operand = operand
        self.targetShape = targetShape
        super.init(name: name, type: operand.type, shape: targetShape)
    }
}

/// Type-cast instruction
/// Casts the element type of a tensor to another
/// - Precondition: The current type must be castable to the target shape
/// - Note: Nope, this is not bitcast
public final class TypeCastInstruction : DefiningInstruction {
    public var operand: Value
    public var targetType: DataType

    public init(name: String, operand: Value, targetType: DataType) {
        self.operand = operand
        self.targetType = targetType
        super.init(name: name, type: targetType, shape: operand.shape)
    }
}

/// Load instruction
/// - Note: Not to be confused with *load* in LLVM. It "loads" or "consumes"
/// an input from the training/inference batch, in terms of neural networks.
public final class LoadInstruction : DefiningInstruction {
    public var source: GlobalValue

    public init(name: String, source: GlobalValue) {
        self.source = source
        super.init(name: name, type: source.type, shape: source.shape)
    }
}

/// Store instruction
/// - Note: Not to be confused with *store* in LLVM. It actually outputs
/// the computed tensor as an output of the neural network. Maybe we
/// want to changed the name to something like "export", but "store" is a
/// nice counterpart of "load".
public final class StoreInstruction : Instruction {
    public var source: Value
    public var destination: GlobalValue

    public init(source: Value, destination: GlobalValue) {
        self.source = source
        self.destination = destination
    }
}
