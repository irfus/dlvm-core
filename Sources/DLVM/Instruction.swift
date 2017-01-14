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

    public func write<Target : TextOutputStream>(to target: inout Target) {
    }

    fileprivate init() {
    }
}

/// Instruction base class
public class DefiningInstruction : Instruction, Value {
    public var name: String
    public var type: DataType

    fileprivate init(name: String, type: DataType) {
        self.name = name
        self.type = type
    }

    public override func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("%\(name) = ")
    }
}

public class NegateInstruction : DefiningInstruction {
    public unowned var operand: Value

    public init(name: String, operand: Value) {
        self.operand = operand
        super.init(name: name, type: operand.type)
    }

    public override func write<Target : TextOutputStream>(to target: inout Target) {
        super.write(to: &target)
        target.write("neg \(operand)")
    }
}

public class ArithmeticInstruction : DefiningInstruction {
    public var `operator`: ArithmeticOperator
    public unowned var leftOperand, rightOperand: Value

    public init(name: String, operator: ArithmeticOperator,
                leftOperand: Value, rightOperand: Value) {
        self.operator = `operator`
        self.leftOperand = leftOperand
        self.rightOperand = rightOperand
        super.init(name: name, type: leftOperand.type)
    }

    public override func write<Target : TextOutputStream>(to target: inout Target) {
        super.write(to: &target)
        target.write("\(`operator`) \(leftOperand) \(rightOperand)")
    }
}

public class ComparisonInstruction : DefiningInstruction {
    public var predicate: ComparisonPredicate
    public unowned var leftOperand, rightOperand: Value

    public init(name: String, predicate: ComparisonPredicate,
                leftOperand: Value, rightOperand: Value) {
        self.predicate = predicate
        self.leftOperand = leftOperand
        self.rightOperand = rightOperand
        super.init(name: name, type: ScalarType.bool)
    }

    public override func write<Target : TextOutputStream>(to target: inout Target) {
        super.write(to: &target)
        target.write("\(predicate) \(leftOperand) \(rightOperand)")
    }
}

public class TensorProductInstruction : DefiningInstruction {
    public unowned var leftOperand, rightOperand: Value

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

    public override func write<Target : TextOutputStream>(to target: inout Target) {
        super.write(to: &target)
        target.write("tmul \(leftOperand) \(rightOperand)")
    }
}

public class ConcatenationInstruction : DefiningInstruction {
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

    public override func write<Target : TextOutputStream>(to target: inout Target) {
        super.write(to: &target)
        target.write("concat \(operands.map{"\($0)"}.joined(separator: ", "))")
    }
}

public class ElementwiseCallInstruction : DefiningInstruction {
    public var function: ElementwiseFunction
    public unowned var operand: Value

    public init(name: String, function: ElementwiseFunction, operand: Value) {
        self.function = function
        self.operand = operand
        super.init(name: name, type: operand.type)
    }

    public override func write<Target : TextOutputStream>(to target: inout Target) {
        super.write(to: &target)
        target.write("\(function) \(operand)")
    }
}

public class AggregateCallInstruction : DefiningInstruction {
    public var function: AggregateFunction
    public unowned var operand: Value

    public init(name: String, function: AggregateFunction, operand: Value) {
        self.function = function
        self.operand = operand
        super.init(name: name, type: operand.type)
    }

    public override func write<Target : TextOutputStream>(to target: inout Target) {
        super.write(to: &target)
        target.write("\(function) \(operand)")
    }
}

public class ShapeCastInstruction : DefiningInstruction {
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

    public override func write<Target : TextOutputStream>(to target: inout Target) {
        super.write(to: &target)
        target.write("shapecast \(operand) to \(targetShape)")
    }
}

public class StoreInstruction<T : GlobalValue> : Instruction {
    public var source: DefiningInstruction
    public var destination: T

    public init(source: DefiningInstruction, destination: T) {
        self.source = source
        self.destination = destination
    }

    public func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("store \(source) to \(destination)")
    }
}
