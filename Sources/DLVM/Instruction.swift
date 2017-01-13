//
//  Instruction.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

//open class Instruction : Value, IRObject {
//    public typealias Parent = BasicBlock
//    public enum ComparisonOperator {
//        case lt, leq, gt, geq, eq, neq
//    }
//    public enum ArithmeticOperator {
//        case add, sub, mul, div, min, max
//    }
//    public enum ActivationFunction {
//        case sigmoid, relu, tanh
//    }
//    public enum TransformationFunction {
//        case log, softmax
//    }
//    public enum Kind {
//        case negate(Value)
//        case arithOp(ArithmeticOperator, Operand, Operand)
//        case compare(ComparisonOperator, Operand, Operand)
//        case dotProduct(TensorVariable, TensorVariable)
//        case product(TensorVariable, TensorVariable)
//        case activation(ActivationFunction, TensorVariable)
//        case transformation(TransformationFunction, TensorVariable)
//        case concat([TensorVariable], dimension: Int)
//        case phi([VariableOperand])
//        case shapeCast(TensorShape, TensorVariable)
//        case condBranch(Operand, then: BasicBlock, else: BasicBlock)
//        case uncondBranch(BasicBlock)
//        case output(TensorVariable)
//    }
//    public let kind: Kind
//    public internal(set) weak var variable: VariableOperand?
//    public internal(set) weak var parent: BasicBlock?
//
//    /// Initialize a standalone instruction by specifying its kind
//    ///
//    /// - Parameter kind: kind of instruction
//    public init(kind: Kind) {
//        self.kind = kind
//    }
//}

public enum ComparisonPredicate {
    case lessThan, lessThanOrEqualTo, greaterThan, greaterThanOrEqualTo, equalTo, notEqualTo
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

public class Instruction : Value, IRObject {
    public var name: String?
    public var type: DataType
    public var shape: TensorShape?
    public weak var parent: BasicBlock?

    fileprivate init(name: String? = nil, type: DataType,
                     shape: TensorShape?, parent: BasicBlock? = nil) {
        self.name = name
        self.type = type
        self.shape = shape
        self.parent = parent
    }

    public func write<Target : TextOutputStream>(to target: inout Target) {
        if let name = name {
            target.write("%\(name) = ")
        }
    }
}

public class NegateInstruction : Instruction {
    public unowned var operand: Value

    public init(operand: Value) {
        self.operand = operand
        super.init(type: operand.type, shape: nil)
    }

    public override func write<Target : TextOutputStream>(to target: inout Target) {
        super.write(to: &target)
        target.write("neg \(operand)")
    }
}

public class ArithmeticInstruction : Instruction {
    public var `operator`: ArithmeticOperator
    public unowned var leftOperand, rightOperand: Value

    public init(operator: ArithmeticOperator, leftOperand: Value, rightOperand: Value) {
        self.operator = `operator`
        self.leftOperand = leftOperand
        self.rightOperand = rightOperand
        super.init(type: leftOperand.type, shape: nil)
    }

    public override func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("\(`operator`) \(leftOperand) \(rightOperand)")
    }
}

public class ComparisonInstruction : Instruction {
    public var predicate: ComparisonPredicate
    public unowned var leftOperand, rightOperand: Value

    public init(predicate: ComparisonPredicate, leftOperand: Value, rightOperand: Value) {
        self.predicate = predicate
        self.leftOperand = leftOperand
        self.rightOperand = rightOperand
        super.init(type: .bool, shape: nil)
    }

    public override func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("\(predicate) \(leftOperand) \(rightOperand)")
    }
}

public class TensorProductInstruction : Instruction {
    public unowned var leftOperand, rightOperand: Value

    public init(leftOperand: Value, rightOperand: Value) {
        self.leftOperand = leftOperand
        self.rightOperand = rightOperand
        super.init(type: .bool, shape: nil)
    }

    public override func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("tmul \(leftOperand) \(rightOperand)")
    }
}

public class ConcatenationInstruction : Instruction {
    public var operands: [Value]
    public var axis: Int

    public init(operands: [Value], axis: Int) {
        precondition(!operands.isEmpty)
        self.operands = Array(operands)
        self.axis = axis
        let firstShape = operands.first?.shape
        let shape = operands.dropFirst().reduce(firstShape, { acc, x in
            x.shape.flatMap {
                acc?.concatenating(with: $0, alongDimension: axis)
            }
        })
        super.init(type: operands[0].type, shape: shape)
    }

    public override func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("concat \(operands.map{"\($0)"}.joined(separator: ", "))")
    }
}

public class ElementwiseCallInstruction : Instruction {
    public var function: ElementwiseFunction
    public unowned var operand: Value

    public init(function: ElementwiseFunction, operand: Value) {
        self.function = function
        self.operand = operand
        super.init(type: .bool, shape: operand.shape)
    }

    public override func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("\(function) \(operand)")
    }
}

public class ShapeCastInstruction : Instruction {
    public var operand: Value
    public var targetShape: TensorShape

    public init(operand: Value, targetShape: TensorShape) {
        self.operand = operand
        self.targetShape = targetShape
        super.init(type: operand.type, shape: targetShape)
    }

    public override func write<Target : TextOutputStream>(to target: inout Target) {
        target.write("shapecast \(operand) to \(targetShape)")
    }
}
