//
//  AST.swift
//  DLVM
//
//  Created by Richard Wei on 12/20/16.
//
//

import enum DLVM.DataType

public enum Macro {
    case type(DataType)
}

public enum Variable {
    case simple(String)
    case recurrent(String, timestep: String, offset: Int)

    var name: String {
        switch self {
        case let .simple(name), let .recurrent(name, _, _):
            return name
        }
    }
}

public enum Role {
    case input, output, hidden, parameter
}

public enum Constant {
    case int(Int)
    case float(Float)
}

public indirect enum Expression {
    /// Integer
    case constant(Constant)
    /// Random
    case random(Constant, Constant)
    /// Variable
    case variable(Variable)
    /// Intrinsic call
    case call(String, [Expression])
    /// Negation
    case negate(Expression)
    /// Element-wise addition
    case add(Expression, Expression)
    /// Element-wise product
    case mul(Expression, Expression)
    /// Tensor product
    case product(Expression, Expression)
    /// Concatenation
    case concat([Expression])
}

public indirect enum Declaration {
    case assignment(Variable, Role, [Int], Expression?)
    case recurrence(String, [Declaration])
}

public enum Statement {
    case macro(Macro)
    case declaration(Declaration)
}

public struct ProgramTree {
    public let statements: [Statement]
}

