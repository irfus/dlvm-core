//
//  Sema.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//  This file contains type checker and semantic analyzer
//

import struct DLVM.ScalarType
import struct DLVM.TensorShape

import enum DLVM.ElementwiseFunction
import enum DLVM.ComparisonPredicate
import enum DLVM.AggregateFunction
import enum DLVM.ArithmeticOperator
import enum DLVM.ReductionFunction
import enum DLVM.ScanFunction
import enum DLVM.BinaryReductionFunction

public enum SemanticError : Error {
    case dataTypeRedeclared
    case dataTypeUnknown(String)
    case moduleNameRedeclared
    case moduleNameMissing
    case initializerMissing(Variable)
    case initializerUnexpected(Variable)
    case variableRedeclared(Variable)
    case variableUndefined(Variable)
    case randomBoundsTypeMismatch(Expression)
    case notAnInitializer(Expression)
    case constantTypeMismatch(Expression, expected: ScalarType)
    case cannotInferShape(Expression)
    case cannotConcatenate(Expression, TensorShape, TensorShape)
    case cannotFormProduct(Expression, TensorShape, Expression, TensorShape)
    case operatorShapeMismatch(Expression, TensorShape, Expression, TensorShape)
    case cannotReshape(Expression, TensorShape, TensorShape)
    case shapeMismatch(Expression, TensorShape, expected: TensorShape, in: Variable)
    case attributeNotOnTop(Attribute)
    case argumentCountMismatch(Expression, count: Int, expected: Int)
    case functionUnknown(Expression, String)
    case inputMissing
    case outputMissing
}

extension SemanticError : CustomStringConvertible {
    public var description: String {
        switch self {
        case .dataTypeRedeclared:
            return "Data type is redeclared"
        case let .dataTypeUnknown(typeName):
            return "Data type '\(typeName)' is not a known"
        case .moduleNameRedeclared:
            return "Network name is redeclared"
        case .moduleNameMissing:
            return "Network name is undefined. Would you like to add a '@name ...'?"
        case let .initializerMissing(variable):
            return "Variable \(variable) needs an initializer"
        case let .initializerUnexpected(variable):
            return "Variable \(variable) should not have an initializer"
        case let .variableRedeclared(variable):
            return "Variable \(variable) is redeclared"
        case let .variableUndefined(variable):
            return "Variable \(variable) is undefined"
        case let .randomBoundsTypeMismatch(expr):
            return "Operands of the random expression do not have the same type \(expr)"
        case let .notAnInitializer(expr):
            return "Expression \(expr) is not an initializer expression"
        case let .constantTypeMismatch(expr, expected: type):
            return "The type of the constant \(expr) does not match the data type of the network (\(type))"
        case let .cannotInferShape(expr):
            return "Cannot infer the shape of expresison \(expr)"
        case let .cannotConcatenate(expr, shape1, shape2):
            return "Cannot perform concatenation \(expr) on shapes \(shape1) and \(shape2)"
        case let .cannotFormProduct(lhsExpr, lhsShape, rhsExpr, rhsShape):
            return "Cannot form product between \(lhsExpr) of shape \(lhsShape) and \(rhsExpr) of shape \(rhsShape)"
        case let .operatorShapeMismatch(lhs, lShape, rhs, rShape):
            return "The shape of the left-hand side \(lhs) (\(lShape)) does not match the shape of right-hand side \(rhs) (\(rShape))"
        case let .cannotReshape(expr, shape, targetShape):
            return "Experssion \(expr) of shape \(shape) cannot be reshaped to \(targetShape) due to mismatch in contiguous memory size"
        case let .shapeMismatch(expr, shape, expected: expectedShape, in: variable):
            return "Expression \(expr) of shape \(shape) does not match the expected shape \(expectedShape) of variable \(variable)"
        case let .attributeNotOnTop(attr):
            return "Attribute \(attr) is not placed on the top of declarations"
        case let .argumentCountMismatch(expr, count: count, expected: expectedCount):
            return "Function call \(expr) has \(count) arguments, but \(expectedCount) are expected"
        case let .functionUnknown(expr, funcName):
            return "Unknown function name '\(funcName)' is found in expression \(expr)"
        case .inputMissing:
            return "I can't find any input layer"
        case .outputMissing:
            return "I can't find any output layer"
        }
    }
}

public protocol Node {
    var name: String { get }
    var shape: TensorShape { get }
}

/// Parameter (param[xxx])
public struct Parameter : Node {
    public enum Initializer {
        public enum Value {
            case int(Int)
            case float(Double)
        }
        case constant(Value)
        case random(Value, Value)
    }
    public let name: String
    public let shape: TensorShape
    public let initializer: Initializer
}

/// Input (in[])
public struct Input : Node {
    public let name: String
    public let shape: TensorShape
}

/// Layer (hidden[])
public struct Layer : Node {
    public let name: String
    public let shape: TensorShape
    public let expression: Expression
    public let isOutput: Bool
}

struct RecurrenceContext {
    let timeStep: String
    let shapes: [String : TensorShape]
}

/// Environment for semantics analysis
struct TypeEnvironment {

    public var moduleName: String?

    /// Symbol table of nodes
    private var nodes: [String : Node] = [:]

    private(set) var parameters: [Parameter] = []
    private(set) var inputs: [Input] = []
    private(set) var layers: [Layer] = []
    
    /// Default data type: float32
    var dataType: ScalarType = .float(32) {
        didSet {
            isCustomDataType = true
        }
    }
    
    private(set) var isCustomDataType = false

    private var recurrences: [RecurrenceContext] = []

    private mutating func register(_ node: Node) {
        nodes[node.name] = node
    }

    mutating func insert(_ input: Input) {
        register(input)
        inputs.append(input)
    }

    mutating func insert(_ parameter: Parameter) {
        register(parameter)
        parameters.append(parameter)
    }

    mutating func insert(_ layer: Layer) {
        register(layer)
        layers.append(layer)
    }

    mutating func pushRecurrence(_ recurrence: RecurrenceContext) {
        recurrences.append(recurrence)
    }

    mutating func popRecurrence() {
        recurrences.removeLast()
    }

    var inRecurrence: Bool {
        return !recurrences.isEmpty
    }

    var isEmpty: Bool {
        return nodes.isEmpty
    }

    func contains(_ key: String) -> Bool {
        return nodes.keys.contains(key)
    }

    subscript(key: String) -> TensorShape? {
        get {
            let lookup = nodes[key]?.shape
            /// If not in recurrence or lookup is available, return
            if !inRecurrence || lookup != nil {
                return lookup
            }
            /// Otherwise, look through all recurrence contexts for
            /// the symbol
            for recCtx in recurrences.reversed() {
                if let shape = recCtx.shapes[key] {
                    return shape
                }
            }
            return nil
        }
    }
}

extension ScalarType {
    init?(name: String) {
        switch name {
        case "int8": self = .int(8)
        case "int16": self = .int(16)
        case "int32": self = .int(32)
        case "int64": self = .int(64)
        case "float8": self = .float(8)
        case "float16": self = .float(16)
        case "float32": self = .float(32)
        case "float64": self = .float(64)
        default: return nil
        }
    }
}

/// Program semantics
/// TODO: support recurrence
public class Program {

    public var moduleName: String
    
    /// Default type: float32
    public internal(set) var dataType: ScalarType = .float(32)
    
    public internal(set) var inputs: [Input] = []
    public internal(set) var layers: [Layer] = []
    public internal(set) var parameters: [Parameter] = []

    public init(parse: ProgramTree) throws {
        /// Create a type environment for semantic analysis
        var env = TypeEnvironment()
        /// Type-check
        try Program.check(parse, in: &env)
        /// Check existence of proper declarations
        guard !env.inputs.isEmpty else {
            throw SemanticError.inputMissing
        }
        guard env.layers.contains(where: {$0.isOutput}) else {
            throw SemanticError.outputMissing
        }
        guard let moduleName = env.moduleName else {
            throw SemanticError.moduleNameMissing
        }
        /// Initialize properties
        self.inputs = env.inputs
        self.layers = env.layers
        self.parameters = env.parameters
        self.moduleName = moduleName
    }

    static func check(_ parse: ProgramTree, in env: inout TypeEnvironment) throws {
        for stmt in parse.statements {
            try Program.check(stmt, in: &env)
        }
    }

    /// Check statement
    /// - Throws: SemanticError
    static func check(_ statement: Statement, in env: inout TypeEnvironment) throws {
        switch statement {
        /// Type macro
        case let .attribute(macro):
            try check(macro, in: &env)
        /// Declaration
        case let .declaration(decl):
            try check(decl, in: &env)
        }
    }

    /// Check macro
    /// - Throws: SemanticError
    static func check(_ macro: Attribute, in env: inout TypeEnvironment) throws {
        guard env.isEmpty else {
            throw SemanticError.attributeNotOnTop(macro)
        }
        switch macro {
        case let .type(typeName):
            guard !env.isCustomDataType else {
                throw SemanticError.dataTypeRedeclared
            }
            guard let type = ScalarType(name: typeName) else {
                throw SemanticError.dataTypeUnknown(typeName)
            }
            env.dataType = type
        case let .name(name):
            guard env.moduleName == nil else {
                throw SemanticError.moduleNameRedeclared
            }
            env.moduleName = name
        }
    }
    
    /// Check declaration
    /// - Throws: SemanticError
    static func check(_ declaration: Declaration, in env: inout TypeEnvironment) throws {
        switch declaration {
            
        /// ## Grand sanity check begin ##
        
        /// Check for redeclaration
        case let .assignment(variable, _, _, _)
            where env.contains(variable.name):
            throw SemanticError.variableRedeclared(variable)
            
        /// If declaration is input layer with an init expr assigned
        /// to it, erorr
        case let .assignment(variable, .input, _, _?):
            throw SemanticError.initializerUnexpected(variable)

        /// No init expr for a non-input node, error
        case let .assignment(variable, .output, _, nil),
             let .assignment(variable, .hidden, _, nil),
             let .assignment(variable, .parameter, _, nil):
            throw SemanticError.initializerMissing(variable)

        /// ## Grand environment filling begin ##
            
        /// Input
        case let .assignment(variable, .input, shapeComponents, nil):
            let shape = TensorShape(shapeComponents)
            let input = Input(name: variable.name, shape: shape)
            env.insert(input)

        /// Parameter
        case let .assignment(variable, .parameter, shapeComponents, expr?):
            let shape = TensorShape(shapeComponents)

            let initializer: Parameter.Initializer

            /// Initializer
            switch expr {

            /// Constant type mismatch
            case .constant(.int(_)) where env.dataType.base != .int,
                 .constant(.float(_)) where env.dataType.base != .float:
                throw SemanticError.constantTypeMismatch(expr, expected: env.dataType)

            /// Int constant
            case let .constant(.int(i)):
                initializer = .constant(.int(i))

            /// Float constant
            case let .constant(.float(f)):
                initializer = .constant(.float(f))

            /// Random int initializer
            case let .random(.int(lo), .int(hi)):
                initializer = .random(.int(lo), .int(hi))

            /// Random float initializer
            case let .random(.float(lo), .float(hi)):
                initializer = .random(.float(lo), .float(hi))

            /// Non-matching types of random bounds
            case .random(_, _):
                throw SemanticError.randomBoundsTypeMismatch(expr)

            default:
                throw SemanticError.notAnInitializer(expr)
            }
            
            let parameter = Parameter(name: variable.name,
                                      shape: shape,
                                      initializer: initializer)
            env.insert(parameter)

        /// ## Grand deep check begin ##

        case let .assignment(variable, .hidden, shapeComponents, expr?):
            let shape = TensorShape(shapeComponents)
            try check(expr, variable: variable, expectedShape: shape, in: &env)
            let layer = Layer(name: variable.name,
                              shape: shape,
                              expression: expr,
                              isOutput: false)
            env.insert(layer)

        case let .assignment(variable, .output, shapeComponents, expr?):
            let shape = TensorShape(shapeComponents)
            try check(expr, variable: variable, expectedShape: shape, in: &env)
            let output = Layer(name: variable.name,
                               shape: shape,
                               expression: expr,
                               isOutput: true)
            env.insert(output)

        case let .recurrence(timeStep, decls):
            /// Recurrent time step ('t', for example) is bound only within the
            /// recurrence. Unlike time step, declarations in a recurrence are
            /// **globally bound**.
            /// Recurrence allows circular dependencies. To do this, we push
            /// the enclosing declarations as a 'recurrence context' into env
            /// before checking inner-scope declarations in normal order.
            /// Create a recurrent context containing a time step and a symbol
            /// table of shapes
            var contextShapes: [String : TensorShape] = [:]
            for case let .assignment(variable, _, shapeComponents, _) in decls {
                guard !env.contains(variable.name) else {
                    throw SemanticError.variableRedeclared(variable)
                }
                contextShapes[variable.name] = TensorShape(shapeComponents)
            }
            let context = RecurrenceContext(timeStep: timeStep, shapes: contextShapes)
            /// Push recurrent context for inner scope
            env.pushRecurrence(context)
            for decl in decls {
                try check(decl, in: &env)
            }
            /// Pop recurrent context to parent scope
            env.popRecurrence()
        }
    }

    /// Check expression
    /// - Throws: SemanticError
    static func check(_ expression: Expression, variable: Variable,
                      expectedShape: TensorShape, in env: inout TypeEnvironment) throws {
        let shape = try Program.shape(of: expression, in: &env)
        guard shape == expectedShape else {
            throw SemanticError.shapeMismatch(expression, shape,
                                              expected: expectedShape,
                                              in: variable)
        }
    }

    /// Infer type of expression
    /// - Throws: SemanticError
    static func shape(of expression: Expression,
                      in env: inout TypeEnvironment) throws -> TensorShape {
        switch expression {

        case let .variable(v):
            guard let shape = env[v.name] else {
                throw SemanticError.variableUndefined(v)
            }
            return shape

        case let .infixOp(_, sideExpr, .constant(const)),
             let .infixOp(_, .constant(const), sideExpr):
            let sideShape = try shape(of: sideExpr, in: &env)
            /// If a float constant is used under an int context, error
            if case .float(_) = const, env.dataType.base == .int {
                throw SemanticError.cannotInferShape(expression)
            }
            return sideShape

        case let .infixOp(_, lhs, rhs):
            let leftShape = try shape(of: lhs, in: &env)
            let rightShape = try shape(of: rhs, in: &env)
            guard leftShape == rightShape else {
                throw SemanticError.operatorShapeMismatch(lhs, leftShape, rhs, rightShape)
            }
            return leftShape

        case let .negate(e):
            return try shape(of: e, in: &env)

        case let .concat(exprs, dimension: dim):
            precondition(!exprs.isEmpty) // Not possible
            let firstShape = try shape(of: exprs[0], in: &env)
            return try exprs.dropFirst().reduce(firstShape) { acc, expr in
                let nextShape = try shape(of: expr, in: &env)
                guard let newShape = acc.concatenating(
                    with: nextShape, alongDimension: dim) else {
                    throw SemanticError.cannotConcatenate(expr, acc, nextShape)
                }
                return newShape
            }

        case let .reshape(expr, shape: dims):
            let exprShape = try shape(of: expr, in: &env)
            let exprSize = exprShape.contiguousSize
            let targetShape = TensorShape(dims)
            let targetSize = targetShape.contiguousSize
            guard exprSize == targetSize else {
                throw SemanticError.cannotReshape(expr, exprShape, targetShape)
            }
            return TensorShape(dims)

        case let .product(lhs, rhs):
            let lhsShape = try shape(of: lhs, in: &env)
            let rhsShape = try shape(of: rhs, in: &env)
            guard let prodShape = lhsShape.matrixMultiplied(by: rhsShape) else {
                throw SemanticError.cannotFormProduct( lhs, lhsShape, rhs, rhsShape)
            }
            return prodShape

        /// For now we assume only unary and binary functions
        case let .call(funcName, args) where args.count == 1:
            let argShape = try shape(of: args[0], in: &env)
            if ScanFunction.lexicon.keys.contains(funcName) ||
                ElementwiseFunction.lexicon.keys.contains(funcName) ||
                AggregateFunction.lexicon.keys.contains(funcName) {
                return argShape
            }
            if ReductionFunction.lexicon.keys.contains(funcName) {
                return [] // Rank 0, scalar
            }
            throw SemanticError.argumentCountMismatch(expression,
                                                      count: args.count,
                                                      expected: 1)

        case let .call(funcName, args) where args.count == 2:
            let firstArgShape = try shape(of: args[0], in: &env)
            let secondArgShape = try shape(of: args[1], in: &env)
            guard firstArgShape == secondArgShape else {
                throw SemanticError.operatorShapeMismatch(args[0], firstArgShape, args[1], secondArgShape)
            }
            if BinaryReductionFunction.lexicon.keys.contains(funcName) {
                return []
            }
            if ArithmeticOperator.lexicon.keys.contains(funcName) {
                return firstArgShape
            }
            throw SemanticError.argumentCountMismatch(expression,
                                                      count: args.count,
                                                      expected: 2)

        case let .call(funcName, _):
            throw SemanticError.functionUnknown(expression, funcName)
            
        default:
            throw SemanticError.cannotInferShape(expression)
            
        }
    }
    
}
