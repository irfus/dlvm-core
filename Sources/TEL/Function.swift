//
//  Function.swift
//  DLVM
//
//  Created by Richard Wei on 1/29/17.
//
//

import DLVM

public struct Function {
    public var name: String
    public var type: FunctionType
    public var instructionType: DefiningInstruction.Type
}

/// - MARK: Helper factory functions
extension Function {
    
    static func aggregation(_ name: String, function: AggregationFunction) -> Function {
        return Function(name: name, type: .homomorphicUnary, instructionType: AggregationInstruction.self)
    }

    static func elementwise(_ name: String, function: ElementwiseFunction) -> Function {
        return Function(name: name, type: .homomorphicUnary, instructionType: ElementwiseInstruction.self)
    }

    static func arithmetic(_ name: String, operator: ArithmeticOperator) -> Function {
        return Function(name: name, type: .homomorphicBinary, instructionType: ArithmeticInstruction.self)
    }
    
    static func logic(_ name: String, operator: LogicOperator) -> Function {
        return Function(name: name, type: .homomorphicBinary, instructionType: LogicInstruction.self)
    }
    
    static func comparison(_ name: String, predicate: ComparisonPredicate) -> Function {
        return Function(name: name, type: .comparison, instructionType: ComparisonInstruction.self)
    }
    
    static func binaryReduction(_ name: String, function: BinaryIntegrationFunction) -> Function {
        return Function(name: name, type: .binaryReduction, instructionType: BinaryReductionInstruction.self)
    }

    static func matrixMultiplication(_ name: String) -> Function {
        return Function(name: name, type: .matrixMultiplication, instructionType: MatrixMultiplicationInstruction.self)
    }

    static func tensorMultiplication(_ name: String) -> Function {
        return Function(name: name, type: .tensorMultiplication, instructionType: TensorMultiplicationInstruction.self)
    }
    
}

public let builtinFunctions: [Function] = [
    /// Elementwise tranfer functions
    .elementwise("log", function: .log),
    .elementwise("relu", function: .relu),
    .elementwise("exp", function: .exp),
    .elementwise("tan", function: .tan),
    .elementwise("tanh", function: .tanh),
    .elementwise("atan", function: .atan),
    .elementwise("sin", function: .sin),
    .elementwise("asin", function: .asin),
    .elementwise("cos", function: .cos),
    .elementwise("acos", function: .acos),
    .elementwise("sqrt", function: .sqrt),
    .elementwise("ceil", function: .ceil),
    .elementwise("floor", function: .floor),

    /// Aggregate transfer functions
    .aggregation("softmax", function: .softmax),
    .aggregation("logSoftmax", function: .logSoftmax),

    /// Logical operators
    .logic("&&", operator: .and),
    .logic("||", operator: .or),
    .logic("^^", operator: .xor),

    /// Arithmetic operators
    .arithmetic("+", operator: .add),
    .arithmetic("-", operator: .subtract),
    .arithmetic("*", operator: .multiply),
    .arithmetic("/", operator: .divide),
    .arithmetic("min", operator: .min),
    .arithmetic("max", operator: .max),
    .arithmetic("pow", operator: .power),
    .arithmetic("mod", operator: .modulo),

    /// Comparison operators
    .comparison(">", predicate: .greaterThan),
    .comparison("<", predicate: .lessThan),
    .comparison(">=", predicate: .greaterThanOrEqualTo),
    .comparison("<=", predicate: .lessThanOrEqualTo),
    .comparison("==", predicate: .equalTo),
    .comparison("!=", predicate: .notEqualTo),

    /// Matrix multiplication operators (equivalent)
    .matrixMultiplication("•"),
    .matrixMultiplication("."),
    .matrixMultiplication("⊗"),

    /// Binary reduction functions
    .binaryReduction("crossEntropy", function: .crossEntropy)
]

public let builtinFunctionTable: [String : Function] = {
    var dict: [String : Function] = [:]
    for function in builtinFunctions {
        dict[function.name] = function
    }
    return dict
}()
