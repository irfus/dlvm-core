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
}

public enum ElementwiseFunction {
    case sigmoid, relu, tanh, log
}

public enum AggregateFunction {
    case softmax
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

public final class NegationInstruction : DefiningInstruction {
    public var operand: Value

    public init(name: String, operand: Value) {
        self.operand = operand
        super.init(name: name, type: operand.type)
    }
}

public final class ArithmeticInstruction : DefiningInstruction {
    public var `operator`: ArithmeticOperator
    public var leftOperand, rightOperand: Value

    public init(name: String, operator: ArithmeticOperator,
                leftOperand: Value, rightOperand: Value) {
        self.operator = `operator`
        self.leftOperand = leftOperand
        self.rightOperand = rightOperand
        super.init(name: name, type: leftOperand.type)
    }
}

public class ComparisonInstruction : DefiningInstruction {
    public var predicate: ComparisonPredicate
    public var leftOperand, rightOperand: Value

    public init(name: String, predicate: ComparisonPredicate,
                leftOperand: Value, rightOperand: Value) {
        self.predicate = predicate
        self.leftOperand = leftOperand
        self.rightOperand = rightOperand
        super.init(name: name, type: ScalarType.bool)
    }
}

public final class TensorProductInstruction : DefiningInstruction {
    public var leftOperand, rightOperand: Value

    public init(name: String, leftOperand: Value, rightOperand: Value) {
        self.leftOperand = leftOperand
        self.rightOperand = rightOperand
        let newType: DataType
        if let lhsType = leftOperand.type as? TensorType,
           let rhsType = rightOperand.type as? TensorType {
            let newShape = (lhsType.shape ⊗ rhsType.shape) ?? lhsType.shape
            newType = TensorType(base: lhsType.base, size: lhsType.size, shape: newShape)
        } else {
            newType = leftOperand.type
        }
        super.init(name: name, type: newType)
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

public final class ElementwiseCallInstruction : DefiningInstruction {
    public var function: ElementwiseFunction
    public var operand: Value

    public init(name: String, function: ElementwiseFunction, operand: Value) {
        self.function = function
        self.operand = operand
        super.init(name: name, type: operand.type)
    }
}

public final class AggregateCallInstruction : DefiningInstruction {
    public var function: AggregateFunction
    public var operand: Value

    public init(name: String, function: AggregateFunction, operand: Value) {
        self.function = function
        self.operand = operand
        super.init(name: name, type: operand.type)
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
