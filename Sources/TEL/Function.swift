//
//  Function.swift
//  DLVM
//
//  Created by Richard Wei on 1/29/17.
//
//

import DLVM

public protocol FunctionProtocol {
    var name: String { get }
    var type: FunctionType { get }
}

public struct Function<IRFunction> : FunctionProtocol {
    typealias Instruction = FunctionCallInstruction<IRFunction>
    public var name: String
    public var type: FunctionType
    public var irFunction: IRFunction
}

public extension FunctionProtocol {
    public func makeInstruction(withArguments args: [Value],
                                using builder: IRBuilder,
                                name: String? = nil) -> DefiningInstruction {
        switch self {
        case let f as Function<AggregationFunction>:
            return builder.makeAggregation(f.irFunction, args[0], name: name)
        case let f as Function<ElementwiseFunction>:
            return builder.makeElementwiseTransformation(f.irFunction, args[0], name: name)
        case let f as Function<ArithmeticOperator>:
            return builder.makeArithmeticOperation(f.irFunction, args[0], args[1], name: name)
        case let f as Function<LogicOperator>:
            return builder.makeLogicOperation(f.irFunction, args[0], args[1], name: name)
        case let f as Function<ComparisonPredicate>:
            return builder.makeComparison(f.irFunction, args[0], args[1], name: name)
        case let f as Function<BinaryIntegrationFunction>:
            return builder.makeBinaryReduction(f.irFunction, args[0], args[1], name: name)
        default:
            preconditionFailure("Unsupported function class \(type(of: self))")
        }
    }
}

/// - MARK: Helper factory functions

private func aggregation(_ name: String, function: AggregationFunction)
    -> Function<AggregationFunction> {
        return Function<AggregationFunction>(name: name, type: .homomorphicUnary, irFunction: function)
}

private func elementwise(_ name: String, function: ElementwiseFunction)
    -> Function<ElementwiseFunction> {
    return Function<ElementwiseFunction>(name: name, type: .homomorphicUnary, irFunction: function)
}

private func arithmetic(_ name: String, operator: ArithmeticOperator)
    -> Function<ArithmeticOperator> {
    return Function<ArithmeticOperator>(name: name, type: .homomorphicBinary, irFunction: `operator`)
}

private func logic(_ name: String, operator: LogicOperator)
    -> Function<LogicOperator> {
    return Function<LogicOperator>(name: name, type: .homomorphicBinary, irFunction: `operator`)
}

private func comparison(_ name: String, predicate: ComparisonPredicate)
    -> Function<ComparisonPredicate> {
    return Function<ComparisonPredicate>(name: name, type: .comparison, irFunction: predicate)
}

private func binaryReduction(_ name: String, function: BinaryIntegrationFunction)
    -> Function<BinaryIntegrationFunction> {
    return Function<BinaryIntegrationFunction>(name: name, type: .binaryReduction, irFunction: function)
}

/// - MARK: TEL built-in functions
public let builtinFunctions: [FunctionProtocol] = [
    /// Elementwise tranfer functions
    elementwise("sigmoid", function: .sigmoid),
    elementwise("log", function: .log),
    elementwise("relu", function: .relu),
    elementwise("exp", function: .exp),
    elementwise("tan", function: .tan),
    elementwise("tanh", function: .tanh),
    elementwise("atan", function: .atan),
    elementwise("sin", function: .sin),
    elementwise("asin", function: .asin),
    elementwise("cos", function: .cos),
    elementwise("acos", function: .acos),
    elementwise("sqrt", function: .sqrt),
    elementwise("ceil", function: .ceil),
    elementwise("floor", function: .floor),

    /// Aggregate transfer functions
    aggregation("softmax", function: .softmax),
    aggregation("logSoftmax", function: .logSoftmax),

    /// Logical operators
    logic("&&", operator: .and),
    logic("||", operator: .or),
    logic("^^", operator: .xor),

    /// Arithmetic operators
    arithmetic("+", operator: .add),
    arithmetic("-", operator: .subtract),
    arithmetic("*", operator: .multiply),
    arithmetic("/", operator: .divide),
    arithmetic("min", operator: .min),
    arithmetic("max", operator: .max),
    arithmetic("pow", operator: .power),
    arithmetic("mod", operator: .modulo),

    /// Comparison operators
    comparison(">", predicate: .greaterThan),
    comparison("<", predicate: .lessThan),
    comparison(">=", predicate: .greaterThanOrEqualTo),
    comparison("<=", predicate: .lessThanOrEqualTo),
    comparison("==", predicate: .equalTo),
    comparison("!=", predicate: .notEqualTo),

    /// Binary reduction functions
    binaryReduction("crossEntropy", function: .crossEntropy)
]

public let builtinFunctionTable: [String : FunctionProtocol] = {
    var dict: [String : FunctionProtocol] = [:]
    for function in builtinFunctions {
        dict[function.name] = function
    }
    return dict
}()
