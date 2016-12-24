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
    case typeMismatch
    case dataTypeRedeclared
    case outputRedeclared(Variable)
    case initializerMissing(Variable)
    case initializerUnexpected(Variable)
    case variableRedeclared(Variable)
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
            case float(Float)
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
    private var shapes: [String : TensorShape] = [:]

    var parameters: [Parameter] = []
    var inputs: [Input] = []
    var layers: [Layer] = []
    var output: Layer?
    var dataType: DataType = .float32
    var isCustomDataType = false

    private var recurrences: [RecurrenceContext] = []

    /// TODO: add accessors for each

    mutating func pushRecurrence(_ recurrence: RecurrenceContext) {
        recurrences.append(recurrence)
    }

    mutating func popRecurrence() {
        recurrences.removeLast()
    }

    func contains(_ key: String) -> Bool {
        return shapes.keys.contains(key)
    }

    subscript(key: String) -> TensorShape? {
        get {
            return shapes[key]
        }
        set {
            shapes[key] = newValue
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

    init(_ parse: ProgramTree) throws {
        /// Create a type environment for semantic analysis
        var env = TypeEnvironment()
        /// Check every statement (top-level items)
        for stmt in parse.statements {
            try Program.check(stmt, in: &env)
        }
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

    /// Check statement
    /// - Throws: SemanticError
    static func check(_ statement: Statement, in env: inout TypeEnvironment) throws {
        switch statement {
        /// Type macro
        case let .macro(.type(type)):
            guard !env.isCustomDataType else {
                throw SemanticError.dataTypeRedeclared
            }
            env.isCustomDataType = true
            env.dataType = type
            
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
            env.inputs.append(input)

        /// Parameter
        case let .assignment(variable, .parameter, shapeComponents, expr?):
            let shape = TensorShape(shapeComponents)
            /// TODO:
            /// 1. Error when `expr` is not float/int/floatRandom/intRandom
            switch expr {
            case .int(_), .float(_):
                // TODO
                break
            default:
                // TODO
                break
            }
            /// 2. Error when `expr`'s type (int/float) does not match env.dataType

            
            /// - note: make use of pattern matching under the same switch
            //  let param = Parameter(name: variable.name, shape: shape, initializer: TODO)
            //  env.parameters.append(param)
            break

        /// ## Grand deep check begin ##
            
        /// Hidden and output checked the same way
        case let .assignment(variable, role, shapeComponents, expr?)
            where role == .hidden || role == .output:
            let shape = TensorShape(shapeComponents)
            try check(expr, for: shape, in: &env)
            env[variable.name] = shape

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

    static func check(_ expression: Expression, for shape: TensorShape,
                      in env: inout TypeEnvironment) throws {
        /// TODO
    }

}
