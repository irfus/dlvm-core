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
    case initializerMissing(Variable)
    case initializerUnexpected(Variable)
}

public protocol Node {
    var name: String { get }
    var shape: TensorShape { get }
}

/// Parameter (param[xxx])
public class Parameter : Node {
    public enum Initializer {
        case int(Int)
        case float(Float)
        case intRandom(Int, Int)
        case floatRandom(Float, Float)
    }
    public let name: String
    public let shape: TensorShape
    public let initializer: Initializer

    public init(name: String, shape: TensorShape, initializer: Initializer) {
        self.name = name
        self.shape = shape
        self.initializer = initializer
    }
}

public class InputLayer : Node {
    public let name: String
    public let shape: TensorShape

    public init(name: String, shape: TensorShape) {
        self.name = name
        self.shape = shape
    }
}

public class OutputLayer : Node {
    public let name: String
    public let shape: TensorShape
    
    /// Add dependency field

    public init(name: String, shape: TensorShape) {
        self.name = name
        self.shape = shape
    }
}

/// Hidden layer (hidden[xxx])
public class HiddenLayer : Node {
    public let name: String
    public let shape: TensorShape
    
    /// Add dependency field

    public init(name: String, shape: TensorShape) {
        self.name = name
        self.shape = shape
    }
}

/// Environment for semantics analysis
/// To be passed to CodeGen
struct TypeEnvironment {
    private var shapes: [String : TensorShape] = [:]

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

    public internal(set) var parameters: [Parameter] = []

    let env = TypeEnvironment()

    init(_ parse: ProgramTree) throws {
        for stmt in parse.statements {
            try typeCheckStatement(stmt, in: &env)
        }
    }

    func typeCheckStatement(_ stmt: Statement, in: inout TypeEnvironment) throws {
        var dataTypeDefined = false

        switch stmt {
        /// Macro
        case let .macro(macro):
            /// Type declaraction
            if case let .type(type) = macro {
                if dataTypeDefined {
                    throw SemanticError.dataTypeRedeclared
                }
                dataTypeDefined = true
                self.dataType = type
            }
        /// Declaration
        case let .declaration(decl):
            switch decl {
            case let .assignment(variable, declType, nil):
                /// If declaration is input layer, expr is required
                if declType.role != .input {
                    throw SemanticError.initializerMissing(variable)
                }
                /// TODO: type-check assignment block w/o init expr
                break
            case let .assignment(variable, declType, expr?):
                /// If declaration is input layer, expr is not needed
                if declType.role == .input {
                    throw SemanticError.initializerUnexpected(variable)
                }
                /// TODO: type-check assignment block w/ init expr
                break
            case let .recurrence(timestep, decls):
                /// TODO: type-check recurrence block
                break
            }
        }
    }
    
}
