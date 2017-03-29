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
import enum DLVM.ElementwiseOp
import enum DLVM.ComparisonOp
import enum DLVM.ArithmeticOp
import enum DLVM.BinaryOp
import enum DLVM.BooleanOp

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
    case constantTypeMismatch(Expression, expected: DataType)
    case cannotInferShape(Expression)
    case cannotConcatenate(Expression, TensorShape, TensorShape)
    case cannotFormProduct(Expression, TensorShape, Expression, TensorShape)
    case operatorShapeMismatch(Expression, TensorShape, Expression, TensorShape)
    case cannotReshape(Expression, TensorShape, TensorShape)
    case cannotTranspose(Expression, TensorShape)
    case shapeMismatch(Expression, TensorShape, expected: TensorShape, in: Variable)
    case attributeNotOnTop(Attribute)
    case functionTypeError(Expression, FunctionTypeError)
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
            return "Network name is undefined. Would you like to add a 'name ...' declaration?"
        case let .initializerMissing(variable):
            return "Variable \(variable) : \(variable.range) needs an initializer"
        case let .initializerUnexpected(variable):
            return "Variable \(variable) : \(variable.range) should not have an initializer"
        case let .variableRedeclared(variable):
            return "Variable \(variable) : \(variable.range) is redeclared"
        case let .variableUndefined(variable):
            return "Variable \(variable) : \(variable.range) is undefined"
        case let .randomBoundsTypeMismatch(expr):
            return "Operands of the random expression do not have the same type \(expr)"
        case let .notAnInitializer(expr):
            return "Expression \(expr) : \(expr.range) is not an initializer expression"
        case let .constantTypeMismatch(expr, expected: type):
            return "The type of the constant \(expr) : \(expr.range) does not match the data type of the network (\(type))"
        case let .cannotInferShape(expr):
            return "Cannot infer the shape of expresison \(expr) : \(expr.range)"
        case let .cannotConcatenate(expr, shape1, shape2):
            return "Cannot perform concatenation \(expr) : \(expr.range) on shapes \(shape1) and \(shape2)"
        case let .cannotFormProduct(lhsExpr, lhsShape, rhsExpr, rhsShape):
            return "Cannot form product between \(lhsExpr) : \(lhsExpr.range) of shape \(lhsShape) and \(rhsExpr) : \(rhsExpr.range) of shape \(rhsShape)"
        case let .operatorShapeMismatch(lhs, lShape, rhs, rShape):
            return "The shape of the left-hand side \(lhs) (\(lShape)) : \(lhs.range) does not match the shape of right-hand side \(rhs) (\(rShape)) : \(rhs.range)"
        case let .cannotReshape(expr, shape, targetShape):
            return "Experssion \(expr) : \(expr.range) of shape \(shape) cannot be reshaped to \(targetShape) due to mismatch in contiguous memory size"
        case let .cannotTranspose(expr, shape):
            return "Expression \(expr) : \(expr.range) of shape \(shape) cannot be transposed"
        case let .shapeMismatch(expr, shape, expected: expectedShape, in: variable):
            return "Expression \(expr) : \(expr.range) of shape \(shape) does not match the expected shape \(expectedShape) of variable \(variable) : \(variable.range)"
        case let .attributeNotOnTop(attr):
            return "Attribute \(attr) is not placed on the top of declarations"
        case let .functionTypeError(expr, error):
            return "In function call \(expr) at \(expr.range): \(error)"
        case let .functionUnknown(expr, funcName):
            return "Unknown function name '\(funcName)' is found in expression \(expr) : \(expr.range)"
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
struct SemaEnvironment {

    public var moduleName: String?

    /// Symbol table of nodes
    private var nodes: [String : Node] = [:]

    private(set) var parameters: [Parameter] = []
    private(set) var inputs: [Input] = []
    private(set) var layers: [Layer] = []
    
    /// Default data type: float32
    var dataType: DataType = .float(.single) {
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

extension DataType {
    init?(name: String) {
        switch name {
        case "int8": self = .int(8)
        case "int16": self = .int(16)
        case "int32": self = .int(32)
        case "int64": self = .int(64)
        case "float16": self = .float(.half)
        case "float32": self = .float(.single)
        case "float64": self = .float(.double)
        default: return nil
        }
    }
}

/// Program semantics
/// TODO: support recurrence
public class Program {

    public var moduleName: String
    
    /// Default type: float32
    public internal(set) var dataType: DataType = .float(.single)
    
    public internal(set) var inputs: [Input] = []
    public internal(set) var layers: [Layer] = []
    public internal(set) var parameters: [Parameter] = []

    public init(parse: ProgramTree) throws {
        /// Create a type environment for semantic analysis
        var env = SemaEnvironment()
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

    static func check(_ parse: ProgramTree, in env: inout SemaEnvironment) throws {
        for stmt in parse.statements {
            try Program.check(stmt, in: &env)
        }
    }

    /// Check statement
    /// - Throws: SemanticError
    static func check(_ statement: Statement, in env: inout SemaEnvironment) throws {
        switch statement {
        /// Type macro
        case let .attribute(macro, _):
            try check(macro, in: &env)
        /// Declaration
        case let .declaration(decl, _):
            try check(decl, in: &env)
        }
    }

    /// Check macro
    /// - Throws: SemanticError
    static func check(_ macro: Attribute, in env: inout SemaEnvironment) throws {
        guard env.isEmpty else {
            throw SemanticError.attributeNotOnTop(macro)
        }
        switch macro {
        case let .type(typeName, _):
            guard !env.isCustomDataType else {
                throw SemanticError.dataTypeRedeclared
            }
            guard let type = DataType(name: typeName) else {
                throw SemanticError.dataTypeUnknown(typeName)
            }
            env.dataType = type
        case let .name(name, _):
            guard env.moduleName == nil else {
                throw SemanticError.moduleNameRedeclared
            }
            env.moduleName = name
        }
    }
    
    /// Check declaration
    /// - Throws: SemanticError
    static func check(_ declaration: Declaration, in env: inout SemaEnvironment) throws {
        switch declaration {
            
        /// ## Grand sanity check begin ##
        
        /// Check for redeclaration
        case let .assignment(variable, _, _, _, _)
            where env.contains(variable.name):
            throw SemanticError.variableRedeclared(variable)
            
        /// If declaration is input layer with an init expr assigned
        /// to it, erorr
        case let .assignment(variable, .input, _, _?, _):
            throw SemanticError.initializerUnexpected(variable)

        /// No init expr for a non-input node, error
        case let .assignment(variable, .output, _, nil, _),
             let .assignment(variable, .hidden, _, nil, _),
             let .assignment(variable, .parameter, _, nil, _):
            throw SemanticError.initializerMissing(variable)

        /// ## Grand environment filling begin ##
            
        /// Input
        case let .assignment(variable, .input, shapeComponents, nil, _):
            let shape = TensorShape(shapeComponents)
            let input = Input(name: variable.name, shape: shape)
            env.insert(input)

        /// Parameter
        case let .assignment(variable, .parameter, shapeComponents, expr?, _):
            let shape = TensorShape(shapeComponents)

            let initializer: Parameter.Initializer

            /// Initializer
            switch expr {

            /// Constant type mismatch
            case .constant(.int(_, _), _) where env.dataType.base != .int,
                 .constant(.float(_, _), _) where env.dataType.base != .float:
                throw SemanticError.constantTypeMismatch(expr, expected: env.dataType)

            /// Int constant
            case let .constant(.int(i, _), _):
                initializer = .constant(.int(i))

            /// Float constant
            case let .constant(.float(f, _), _):
                initializer = .constant(.float(f))

            /// Random int initializer
            case let .random(.int(lo, _), .int(hi, _), _):
                initializer = .random(.int(lo), .int(hi))

            /// Random float initializer
            case let .random(.float(lo, _), .float(hi, _), _):
                initializer = .random(.float(lo), .float(hi))

            /// Non-matching types of random bounds
            case .random(_, _, _):
                throw SemanticError.randomBoundsTypeMismatch(expr)

            default:
                throw SemanticError.notAnInitializer(expr)
            }
            
            let parameter = Parameter(name: variable.name,
                                      shape: shape,
                                      initializer: initializer)
            env.insert(parameter)

        /// ## Grand deep check begin ##

        case let .assignment(variable, .hidden, shapeComponents, expr?, _):
            let shape = TensorShape(shapeComponents)
            try check(expr, variable: variable, expectedShape: shape, in: &env)
            let layer = Layer(name: variable.name,
                              shape: shape,
                              expression: expr,
                              isOutput: false)
            env.insert(layer)

        case let .assignment(variable, .output, shapeComponents, expr?, _):
            let shape = TensorShape(shapeComponents)
            try check(expr, variable: variable, expectedShape: shape, in: &env)
            let output = Layer(name: variable.name,
                               shape: shape,
                               expression: expr,
                               isOutput: true)
            env.insert(output)

        case let .recurrence(timeStep, decls, _):
            /// Recurrent time step ('t', for example) is bound only within the
            /// recurrence. Unlike time step, declarations in a recurrence are
            /// **globally bound**.
            /// Recurrence allows circular dependencies. To do this, we push
            /// the enclosing declarations as a 'recurrence context' into env
            /// before checking inner-scope declarations in normal order.
            /// Create a recurrent context containing a time step and a symbol
            /// table of shapes
            var contextShapes: [String : TensorShape] = [:]
            for case let .assignment(variable, _, shapeComponents, _, _) in decls {
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
                      expectedShape: TensorShape, in env: inout SemaEnvironment) throws {
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
                      in env: inout SemaEnvironment) throws -> TensorShape {
        switch expression {

        case let .constant(const, _):
            /// If a float constant is used under an int context, error
            if case .float(_, _) = const, env.dataType.base == .int {
                throw SemanticError.cannotInferShape(expression)
            }
            return .scalar

        case let .variable(v, _):
            guard let shape = env[v.name] else {
                throw SemanticError.variableUndefined(v)
            }
            return shape

        case let .infixOp(_, lhs, rhs, _):
            let leftShape = try shape(of: lhs, in: &env)
            let rightShape = try shape(of: rhs, in: &env)
            guard let shape = leftShape.broadcasted(to: rightShape)
                           ?? rightShape.broadcasted(to: leftShape) else {
                throw SemanticError.operatorShapeMismatch(lhs, leftShape, rhs, rightShape)
            }
            return shape

        case let .negate(e, _):
            return try shape(of: e, in: &env)

        case let .concat(exprs, dimension: dim, _):
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

        case let .reshape(expr, shape: dims, _):
            let exprShape = try shape(of: expr, in: &env)
            let exprSize = exprShape.contiguousSize
            let targetShape = TensorShape(dims)
            let targetSize = targetShape.contiguousSize
            guard exprSize == targetSize else {
                throw SemanticError.cannotReshape(expr, exprShape, targetShape)
            }
            return TensorShape(dims)

        case let .product(lhs, rhs, _):
            let lhsShape = try shape(of: lhs, in: &env)
            let rhsShape = try shape(of: rhs, in: &env)
            guard let prodShape = lhsShape.matrixMultiplied(with: rhsShape) else {
                throw SemanticError.cannotFormProduct( lhs, lhsShape, rhs, rhsShape)
            }
            return prodShape

        /// For now we assume only unary and binary functions
        case let .call(funcName, args, _):
            let argShapes = try args.map { try shape(of: $0, in: &env) }
            guard let function = intrinsicTable[funcName] else {
                throw SemanticError.functionUnknown(expression, funcName)
            }
            guard function.argumentCount == args.count else {
                throw SemanticError.functionTypeError(
                    expression, .argumentCountMismatch(expected: function.argumentCount, actual: args.count))
            }
            do {
                let resultShape: TensorShape = try function.resultShape(forArguments: argShapes)
                return resultShape
            } catch let error as FunctionTypeError {
                throw SemanticError.functionTypeError(expression, error)
            }

        case let .transpose(expr, _):
            return try shape(of: expr, in: &env).transpose

        default:
            throw SemanticError.cannotInferShape(expression)
            
        }
    }
    
}
