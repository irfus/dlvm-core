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

public enum ArithmeticOperator {
    case add, subtract, multiply, divide, min, max
    case truncateDivide, floorDivide, mod, pow
}

public enum ElementwiseFunction {
    case sigmoid, relu, tanh
    case log, exp, neg, sign, square, sqrt, round, rsqrt, ceil, floor
    case tan, cos, sin, acos, asin, atan
    case lgamma, digamma, erf, erfc, rint
}

public enum ReductionFunction {
    case sum, product, min, max, logicalAnd, logicalOr, mean
}

public enum BinaryReductionFunction {
    case crossEntropy
}

public enum ScanFunction {
    case scanSum, scanProduct
}

public enum AggregateFunction {
    case softmax, logSoftmax
}

public class Instruction : IRObject {
    public weak var parent: BasicBlock?
    fileprivate init() {}
}

/// Instruction base class
public class DefiningInstruction : Instruction, NamedValue {
    public var name: String
    public var type: DataType

    fileprivate init(name: String, type: DataType) {
        self.name = name
        self.type = type
    }
}

public class UnaryInstruction : DefiningInstruction {
    public var operand: Value

    fileprivate init(name: String, type: DataType, operand: Value) {
        self.operand = operand
        super.init(name: name, type: type)
    }
}

public class BinaryInstruction : DefiningInstruction {
    public var firstOperand, secondOperand: Value

    fileprivate init(name: String, type: DataType, firstOperand: Value, secondOperand: Value) {
        self.firstOperand = firstOperand
        self.secondOperand = secondOperand
        super.init(name: name, type: type)
    }
}

public class UnaryCallInstruction<Function> : UnaryInstruction {
    public var function: Function

    public init(name: String, type: DataType, function: Function, operand: Value) {
        self.function = function
        super.init(name: name, type: type, operand: operand)
    }
}

public class BinaryCallInstruction<Function> : BinaryInstruction {
    public var function: Function

    public init(name: String, type: DataType, function: Function,
                firstOperand: Value, secondOperand: Value) {
        self.function = function
        super.init(name: name, type: type,
                   firstOperand: firstOperand, secondOperand: secondOperand)
    }
}

public class ReductionInstruction : UnaryCallInstruction<ReductionFunction> {
    public init(name: String, function: ReductionFunction, operand: Value) {
        super.init(name: name, type: operand.type.scalarType,
                   function: function, operand: operand)
    }
}

public class HomomorphicUnaryInstruction<Function> : UnaryCallInstruction<Function> {
    public init(name: String, function: Function, operand: Value) {
        super.init(name: name, type: operand.type, function: function, operand: operand)
    }
}

public class HomomorphicBinaryInstruction<Function> : BinaryCallInstruction<Function> {
    public init(name: String, function: Function, firstOperand: Value, secondOperand: Value) {
        super.init(name: name, type: firstOperand.type, function: function,
                   firstOperand: firstOperand, secondOperand: secondOperand)
    }
}

public typealias ElementwiseTransformationInstruction = HomomorphicUnaryInstruction<ElementwiseFunction>
public typealias AggregateTransformationInstruction = HomomorphicUnaryInstruction<AggregateFunction>
public typealias BinaryReductionInstruction = HomomorphicBinaryInstruction<BinaryReductionFunction>
public typealias ScanInstruction = HomomorphicUnaryInstruction<ScanFunction>
public typealias ArithmeticInstruction = HomomorphicBinaryInstruction<ArithmeticOperator>

public class ComparisonInstruction : BinaryInstruction {
    public var predicate: ComparisonPredicate

    public init(name: String, predicate: ComparisonPredicate,
                firstOperand: Value, secondOperand: Value) {
        self.predicate = predicate
        super.init(name: name, type: ScalarType.bool,
                   firstOperand: firstOperand, secondOperand: secondOperand)
    }
}

public final class TensorProductInstruction : BinaryInstruction {
    public init(name: String, firstOperand: Value, secondOperand: Value) {
        let newType: DataType
        if let lhsType = firstOperand.type as? TensorType,
           let rhsType = secondOperand.type as? TensorType {
            let newShape = (lhsType.shape ⊗ rhsType.shape) ?? lhsType.shape
            newType = TensorType(base: lhsType.base, size: lhsType.size, shape: newShape)
        } else {
            newType = firstOperand.type
        }
        super.init(name: name, type: newType,
                   firstOperand: firstOperand, secondOperand: secondOperand)
    }
}

public final class MatrixProductInstruction : BinaryInstruction {
    public init(name: String, firstOperand: Value, secondOperand: Value) {
        let newType: DataType
        if let lhsType = firstOperand.type as? TensorType,
           let rhsType = secondOperand.type as? TensorType {
            let newShape = (lhsType.shape ⊗ rhsType.shape) ?? lhsType.shape
            newType = TensorType(base: lhsType.base, size: lhsType.size, shape: newShape)
        } else {
            newType = firstOperand.type
        }
        super.init(name: name, type: newType,
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
        guard let types = operands.map({$0.type}) as? [TensorType] else {
            super.init(name: name, type: operands[0].type)
            return
        }
        let firstShape = types[0].shape
        let newShape = types.dropFirst().reduce(firstShape, { acc, x in
            acc?.concatenating(with: x.shape, alongDimension: axis)
        })
        let newType = newShape.flatMap { shape in
            TensorType(base: types[0].base, size: types[0].size, shape: shape)
        }
        super.init(name: name, type: newType ?? operands[0].type)
    }
}

public final class ShapeCastInstruction : DefiningInstruction {
    public var operand: Value
    public var targetShape: TensorShape

    public init(name: String, operand: Value, targetShape: TensorShape) {
        self.operand = operand
        self.targetShape = targetShape
        let newType = TensorType(base: operand.type.base,
                                 size: operand.type.size,
                                 shape: targetShape)
        super.init(name: name, type: newType)
    }
}

public final class TypeCastInstruction : DefiningInstruction {
    public var operand: Value
    public var targetBase: TypeBase
    public var targetSize: Int
    
    public init(name: String, operand: Value, targetBase: TypeBase, targetSize: Int) {
        self.operand = operand
        self.targetBase = targetBase
        self.targetSize = targetSize
        var newType = operand.type
        newType.base = targetBase
        newType.size = targetSize
        super.init(name: name, type: newType)
    }
}

public final class LoadInstruction : DefiningInstruction {
    public var source: Value

    public init(name: String, source: Value) {
        self.source = source
        super.init(name: name, type: source.type)
    }
}

public final class StoreInstruction : Instruction {
    public var source: Value
    public var destination: Value

    public init(source: Value, destination: Value) {
        self.source = source
        self.destination = destination
    }
}
