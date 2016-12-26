//
//  Sema.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//  This file contains type checker and semantic analyzer
//

import enum DLVM.DataType
import struct DLVM.TensorShape

public enum SemanticError : Error {
    case dataTypeRedeclared
    case dataTypeUnknown(String)
    case outputRedeclared(Variable)
    case initializerMissing(Variable)
    case initializerUnexpected(Variable)
    case variableRedeclared(Variable)
    case variableUndefined(Variable)
    case randomBoundsTypeMismatch(Expression)
    case notAnInitializer(Expression)
    case constantTypeMismatch(Expression, expected: DataType)
    case cannotInferShape(Expression)
    case cannotConcatenate(Expression, TensorShape, TensorShape)
    case cannotFormProduct(Expression, TensorShape, Expression, TensorShape)
    case operatorShapeMismatch(Expression)
    case shapeMismatch(Expression, expected: TensorShape, in: Variable)
    case typeDeclarationNotOnTop(Macro)
    case argumentCountMismatch(Expression, count: Int, expected: Int)
    case functionUnknown(Expression, String)
    case inputMissing
    case outputMissing
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
}

struct RecurrenceContext {
    let timestep: String
    let shapes: [String : TensorShape]
}

/// Environment for semantics analysis
struct TypeEnvironment {

    /// Symbol table of nodes
    private var nodes: [String : Node] = [:]

    private(set) var parameters: [Parameter] = []
    private(set) var inputs: [Input] = []
    private(set) var layers: [Layer] = []
    
    var output: Layer? {
        willSet {
            if let newValue = newValue {
                nodes[newValue.name] = newValue
            }
        }
    }

    /// Default data type: float32
    var dataType: DataType = .float32 {
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

/// Program semantics
public class Program {
    
    /// Default type: float32
    public internal(set) var dataType: DataType = .float32
    
    public internal(set) var inputs: [Input] = []
    public internal(set) var layers: [Layer] = []
    public internal(set) var output: Layer
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
        guard let output = env.output else {
            throw SemanticError.outputMissing
        }
        /// Initialize properties
        self.inputs = env.inputs
        self.layers = env.layers
        self.output = output
        self.parameters = env.parameters
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
        case let .macro(macro):
            switch macro {
            case let .type(typeName):
                guard env.isEmpty else {
                    throw SemanticError.typeDeclarationNotOnTop(macro)
                }
                guard !env.isCustomDataType else {
                    throw SemanticError.dataTypeRedeclared
                }
                guard let type = DataType(rawValue: typeName) else {
                    throw SemanticError.dataTypeUnknown(typeName)
                }
                env.dataType = type
            }
            
        /// Declaration
        case let .declaration(decl):
            try check(decl, in: &env)
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

        /// If more than one output, error
        case let .assignment(variable, .output, _, _)
            where env.contains(variable.name):
            throw SemanticError.outputRedeclared(variable)

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
            case .constant(.int(_)) where !env.dataType.isInt,
                 .constant(.float(_)) where !env.dataType.isFloat:
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
                              expression: expr)
            env.insert(layer)

        case let .assignment(variable, .output, shapeComponents, expr?):
            let shape = TensorShape(shapeComponents)
            try check(expr, variable: variable, expectedShape: shape, in: &env)
            let output = Layer(name: variable.name,
                               shape: shape,
                               expression: expr)
            env.output = output

        case let .recurrence(timestep, decls):
            /// Recurrent timestep ('t', for example) is bound only within the
            /// recurrence. Unlike timestep, declarations in a recurrence are
            /// **globally bound**.
            /// Recurrence allows circular dependencies. To do this, we push
            /// the enclosing declarations as a 'recurrence context' into env
            /// before checking inner-scope declarations in normal order.
            /// Create a recurrent context containing a timestep and a symbol
            /// table of shapes
            var contextShapes: [String : TensorShape] = [:]
            for case let .assignment(variable, _, shapeComponents, _) in decls {
                guard !env.contains(variable.name) else {
                    throw SemanticError.variableRedeclared(variable)
                }
                contextShapes[variable.name] = TensorShape(shapeComponents)
            }
            let context = RecurrenceContext(timestep: timestep, shapes: contextShapes)
            /// Push recurrent context for inner scope
            env.pushRecurrence(context)
            for decl in decls {
                try check(decl, in: &env)
            }
            /// Pop recurrent context to parent scope
            env.popRecurrence()

        default: break
        }
    }

    static func check(_ expression: Expression, variable: Variable,
                      expectedShape: TensorShape, in env: inout TypeEnvironment) throws {
        let shape = try Program.shape(of: expression, in: &env)
        guard shape == expectedShape else {
            throw SemanticError.shapeMismatch(expression,
                                              expected: expectedShape,
                                              in: variable)
        }
    }

    static func shape(of expression: Expression,
                      in env: inout TypeEnvironment) throws -> TensorShape {
        switch expression {

        case let .variable(v):
            guard let shape = env[v.name] else {
                throw SemanticError.variableUndefined(v)
            }
            return shape

        case let .add(sideExpr, .constant(const)),
             let .add(.constant(const), sideExpr),
             let .sub(sideExpr, .constant(const)),
             let .sub(.constant(const), sideExpr),
             let .mul(sideExpr, .constant(const)),
             let .mul(.constant(const), sideExpr):
            let sideShape = try shape(of: sideExpr, in: &env)
            /// If a float constant is used under an int context, error
            if case .float(_) = const, !env.dataType.isInt {
                throw SemanticError.cannotInferShape(expression)
            }
            return sideShape

        case let .add(lhs, rhs),
             let .sub(lhs, rhs),
             let .mul(lhs, rhs):
            let leftShape = try shape(of: lhs, in: &env)
            let rightShape = try shape(of: rhs, in: &env)
            guard leftShape == rightShape else {
                throw SemanticError.operatorShapeMismatch(expression)
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

        case let .product(lhs, rhs):
            let lhsShape = try shape(of: lhs, in: &env)
            let rhsShape = try shape(of: rhs, in: &env)
            guard let prodShape = lhsShape.product(with: rhsShape) else {
                throw SemanticError.cannotFormProduct( lhs, lhsShape, rhs, rhsShape)
            }
            return prodShape

        /// For now we assume only unary functions
        case let .call("sigmoid", args) where args.count == 1,
             let .call("tanh", args) where args.count == 1,
             let .call("relu", args) where args.count == 1,
             let .call("log", args) where args.count == 1,
             let .call("softmax", args) where args.count == 1:
            return try shape(of: args[0], in: &env)

        case let .call(_, args) where args.count != 1:
            throw SemanticError.argumentCountMismatch(expression,
                                                      count: args.count,
                                                      expected: 1)

        case let .call(funcName, _):
            throw SemanticError.functionUnknown(expression, funcName)
            
        default:
            throw SemanticError.cannotInferShape(expression)
            
        }
    }
    
}
