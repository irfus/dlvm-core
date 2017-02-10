//
//  Instruction.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

///
/// ## Functions
///

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
    case truncateDivide, floorDivide, modulo, power, mean
}

public enum ElementwiseFunction {
    case sigmoid, tanh
    case log, exp, neg, sign, square, sqrt, round, rsqrt, ceil, floor
    case tan, cos, sin, acos, asin, atan
    case lgamma, digamma, erf, erfc, rint
}

public enum ReductionFunction {
    case logical(LogicOperator)
    case arithmetic(ArithmeticOperator)
}

public enum AggregationFunction {
    case softmax, logSoftmax, argmax, argmin
    case scan(ReductionFunction)
}

///
/// ## Instruction protocols
///

public protocol Instruction : class, SelfVerifiable {
    var operands: [Value] { get }
    weak var parent: BasicBlock? { get set }
}

// MARK: - Analysis information
extension Instruction {
    func updateUsers() {
        for operand in operands {
            if let usee = operand as? ManagedUsee {
                usee.addUser(self)
            }
        }
    }
}

public protocol DefiningInstruction : Instruction, NamedValue {
}

/// Abstract class for unary instruction
public protocol UnaryOperator {
    var operand: Value { get set }
}

public extension UnaryOperator {
    public var operands: [Value] {
        return [operand]
    }
}

/// Abstract class for binary instruction
public protocol BinaryOperator {
    var firstOperand: Value { get set }
    var secondOperand: Value { get set }
}

public extension BinaryOperator {
    public var operands: [Value] {
        return [firstOperand, secondOperand]
    }
}

public protocol FunctionCallInstruction : DefiningInstruction {
    associatedtype Function
    var function: Function { get set }
}

/// Unary function calls
public typealias UnaryCallInstruction = FunctionCallInstruction & UnaryOperator

/// Binary function calls
public typealias BinaryCallInstruction = FunctionCallInstruction & BinaryOperator

///
/// ## Instructions
///

public class HomomorphicUnaryInstruction<Function> : UnaryCallInstruction {
    public var parent: BasicBlock?
    public var name: String
    public lazy var type: DataType = self.operand.type
    public lazy var shape: TensorShape = self.operand.shape
    public var function: Function
    public var operand: Value
    public internal(set) var users: NamedObjectSet<Instruction> = []

    public init(name: String, function: Function, operand: Value) {
        self.name = name
        self.function = function
        self.operand = operand
    }
}

public class HomomorphicBinaryInstruction<Function> : BinaryCallInstruction {
    public var parent: BasicBlock?
    public var name: String
    public lazy var type: DataType = self.firstOperand.type
    public lazy var shape: TensorShape =
        self.firstOperand.shape.rank > self.secondOperand.shape.rank
      ? self.firstOperand.shape : self.secondOperand.shape
    public var function: Function
    public var firstOperand: Value
    public var secondOperand: Value
    public internal(set) var users: NamedObjectSet<Instruction> = []

    public init(name: String, function: Function, firstOperand: Value, secondOperand: Value) {
        self.name = name
        self.function = function
        /// Broadcasting
        self.firstOperand = firstOperand
        self.secondOperand = secondOperand
    }
}

/// Shape-preserving, type-preserving (homomorphic) transformation instructions
public typealias ElementwiseInstruction = HomomorphicUnaryInstruction<ElementwiseFunction>
public typealias AggregationInstruction = HomomorphicUnaryInstruction<AggregationFunction>
public typealias ArithmeticInstruction = HomomorphicBinaryInstruction<ArithmeticOperator>
public typealias LogicInstruction = HomomorphicBinaryInstruction<LogicOperator>

/// Reduction instruction
/// - Note: We are assuming that reduction instruction takes any tensor and apply
/// the reduction function between elements contiguously in memory.
/// At the end of the day we will want shape-based reduction, e.g. reducing a
/// dimension from the shape. But this is not a priority.
/// Q: Why do we still want this feature? A: TensorFlow has it.
public class ReductionInstruction : UnaryCallInstruction {
    public var parent: BasicBlock?
    public var name: String
    public lazy var type: DataType = self.operand.type
    public lazy var shape: TensorShape =
        self.axis.flatMap(self.operand.shape.droppingDimension) ?? .scalar
    public var function: ReductionFunction
    public var operand: Value
    public var axis: Int?
    public internal(set) var users: NamedObjectSet<Instruction> = []
    
    public init(name: String, function: ReductionFunction, operand: Value, axis: Int?) {
        self.name = name
        self.function = function
        self.operand = operand
        self.axis = axis
    }
}

/// Comparison instruction
/// - shape: [A] -> [A]
/// - type: _ -> Bool
public final class ComparisonInstruction : BinaryCallInstruction {
    public var parent: BasicBlock?
    public var name: String
    public var type: DataType = .bool
    public lazy var shape: TensorShape = self.firstOperand.shape
    public var function: ComparisonPredicate
    public var firstOperand: Value
    public var secondOperand: Value
    public internal(set) var users: NamedObjectSet<Instruction> = []

    public init(name: String, function: ComparisonPredicate,
                firstOperand: Value, secondOperand: Value) {
        self.name = name
        self.function = function
        self.firstOperand = firstOperand
        self.secondOperand = secondOperand
    }
}

/// Generic tensor multiplication instruction (GETT operation)
public final class TensorMultiplicationInstruction : DefiningInstruction, BinaryOperator {
    public var parent: BasicBlock?
    public var name: String
    public lazy var type: DataType = self.firstOperand.type
    public lazy var shape: TensorShape =
        (self.firstOperand.shape ⊗ self.secondOperand.shape) ?? self.firstOperand.shape
    public var firstOperand: Value
    public var secondOperand: Value
    public internal(set) var users: NamedObjectSet<Instruction> = []

    public init(name: String, firstOperand: Value, secondOperand: Value) {
        self.name = name
        self.firstOperand = firstOperand
        self.secondOperand = secondOperand
    }
}

/// Matrix multiplication instruction (GEMM operation)
/// - Note: This only applies to the two inner dimensions
public final class MatrixMultiplicationInstruction : DefiningInstruction, BinaryOperator {
    public var parent: BasicBlock?
    public var name: String
    public var type: DataType
    public lazy var shape: TensorShape = {
        let newShape = self.firstOperand.shape.matrixMultiplied(with: self.secondOperand.shape)
                    ?? self.firstOperand.shape
        return newShape
    }()
    public var firstOperand: Value
    public var secondOperand: Value
    public internal(set) var users: NamedObjectSet<Instruction> = []
    
    public init(name: String, firstOperand: Value, secondOperand: Value) {
        self.name = name
        self.type = firstOperand.type
        self.firstOperand = firstOperand
        self.secondOperand = secondOperand
    }
}

/// Concatenation instruction
/// Concatenates multiple tensors along an axis dimension
public final class ConcatenationInstruction : DefiningInstruction {
    public var parent: BasicBlock?
    public var name: String
    public lazy var type: DataType = self.operands[0].type
    public lazy var shape: TensorShape = {
        let firstShape = self.operands[0].shape
        let newShape = self.operands.dropFirst().reduce(firstShape, { acc, x in
            acc?.concatenating(with: x.shape, alongDimension: self.axis)
        }) ?? firstShape
        return newShape
    }()
    public var operands: [Value]
    public var axis: Int
    public internal(set) var users: NamedObjectSet<Instruction> = []

    public init(name: String, operands: [Value], axis: Int) {
        precondition(!operands.isEmpty)
        self.name = name
        self.operands = operands
        self.axis = axis
    }
}

public protocol CastInstruction : DefiningInstruction, UnaryOperator {
    associatedtype Target
    var target: Target { get set }
    init(name: String, operand: Value, target: Target)
}

/// Shape-cast instruction
/// Casts the shape of a tensor to another
/// - Precondition: Target shape must be of the same contiguous size
/// - Note: We will want to be able to shape-cast between different contiguous
/// sizes (Q: Reason? A: TensorFlow does.). This should be done after we
/// introduce **slicing**-related instructions.
public final class ShapeCastInstruction : CastInstruction {
    public var parent: BasicBlock?
    public var name: String
    public lazy var type: DataType = self.operand.type
    public lazy var shape: TensorShape = self.target
    public var operand: Value
    public var target: TensorShape
    public internal(set) var users: NamedObjectSet<Instruction> = []

    public init(name: String, operand: Value, target: TensorShape) {
        self.name = name
        self.operand = operand
        self.target = target
    }
}

/// Type-cast instruction
/// Casts the element type of a tensor to another
/// - Precondition: The current type must be castable to the target shape
/// - Note: Nope, this is not bitcast
public final class TypeCastInstruction : CastInstruction {
    public var parent: BasicBlock?
    public var name: String
    public lazy var type: DataType = self.target
    public lazy var shape: TensorShape = self.operand.shape
    public var operand: Value
    public var target: DataType
    public internal(set) var users: NamedObjectSet<Instruction> = []

    public init(name: String, operand: Value, target: DataType) {
        self.name = name
        self.operand = operand
        self.target = target
    }
}

/// Load instruction
/// - Note: Not to be confused with *load* in LLVM. It "loads" or "consumes"
/// an input from the training/inference batch, in terms of neural networks.
public final class LoadInstruction : DefiningInstruction {
    public var parent: BasicBlock?
    public var name: String
    public lazy var type: DataType = self.source.type
    public lazy var shape: TensorShape = self.source.shape
    public var source: Input
    public internal(set) var users: NamedObjectSet<Instruction> = []

    public var operands: [Value] {
        return []
    }

    public init(name: String, source: Input) {
        self.name = name
        self.source = source
    }
}

/// Export instruction
public final class ExportInstruction : Instruction {
    public var parent: BasicBlock?
    public var source: Value
    public var destination: Output

    public var operands: [Value] {
        return [source]
    }

    public init(source: Value, destination: Output) {
        self.source = source
        self.destination = destination
    }
}

/// Store instruction
public final class StoreInstruction : Instruction {
    public var parent: BasicBlock?
    public var source: Value
    public var destination: Parameter

    public var operands: [Value] {
        return [source, destination]
    }

    public init(source: Value, destination: Parameter) {
        self.source = source
        self.destination = destination
    }
}

public final class PhiInstruction : DefiningInstruction {
    public var parent: BasicBlock?
    public var name: String
    public lazy var type: DataType = self.incomingValues[0].0.type
    public lazy var shape: TensorShape = self.incomingValues[0].0.shape
    public var incomingValues: [(Value, BasicBlock)]
    public internal(set) var users: NamedObjectSet<Instruction> = []

    public var operands: [Value] {
        return incomingValues.map{$0.0}
    }

    public init(name: String, incomingValues: [(Value, BasicBlock)]) {
        self.name = name
        self.incomingValues = incomingValues
    }
}

public protocol TerminatorInstruction : Instruction {
    var destinations: [BasicBlock] { get }
}

public final class BranchInstruction : TerminatorInstruction {
    public var parent: BasicBlock?
    public var destination: BasicBlock

    public var operands: [Value] {
        return []
    }

    public var destinations: [BasicBlock] {
        return [destination]
    }

    public init(destination: BasicBlock) {
        self.destination = destination
    }
}
