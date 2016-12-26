//
//  AST.swift
//  DLVM
//
//  Created by Richard Wei on 12/20/16.
//
//

public enum Macro {
    case type(String)
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
    case float(Double)
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
    /// Element-wise subtraction
    case sub(Expression, Expression)
    /// Element-wise product
    case mul(Expression, Expression)
    /// Tensor product
    case product(Expression, Expression)
    /// Concatenation
    case concat([Expression], dimension: Int)
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

extension Constant : CustomStringConvertible {
    public var description: String {
        switch self {
        case let .int(i):   return i.description
        case let .float(f): return f.description
        }
    }
}

extension Variable : CustomStringConvertible {

    public var description: String {
        switch self {
        case let .simple(name): return name
        case let .recurrent(name, timestep: t, offset: offset):
            return name + "[" + t + offset.description + "]"
        }
    }
    
}

extension Expression : CustomStringConvertible {

    public var description: String {
        switch self {
        case let .constant(c): return c.description
        case let .random(lhs, rhs): return "(random \(lhs) \(rhs))"
        case let .variable(v): return v.name
        case let .call(name, args):
            return "(\(name) \(args.map{$0.description}.joined(separator: " ")))"
        case let .negate(expr):
            return "(- \(expr))"
        case let .add(lhs, rhs):
            return "(+ \(lhs) \(rhs))"
        case let .sub(lhs, rhs):
            return "(- \(lhs) \(rhs))"
        case let .mul(lhs, rhs):
            return "(* \(lhs) \(rhs))"
        case let .product(lhs, rhs):
            return "(⊗ \(lhs) \(rhs))"
        case let .concat(exprs, dim):
            return "(concat \(dim) \(exprs.map{$0.description}.joined(separator: " ")))"
        }
    }
    
}

extension Declaration : CustomStringConvertible {

    public var description: String {
        switch self {
        case let .assignment(v, role, dim, expr):
            return "(define \(v):\(role)[\(dim.map{$0.description}.joined(separator: "x"))]\(expr == nil ? "" : " " + expr!.description))"
        case let .recurrence(timestep, decls):
            return "(recurrent \(timestep) \(decls.map{$0.description}.joined(separator: " ")))"
        }
    }
    
}

extension Macro : CustomStringConvertible {

    public var description: String {
        switch self {
        case let .type(t): return "(define-type \(t))"
        }
    }

}

extension Statement : CustomStringConvertible {

    public var description: String {
        switch self {
        case let .macro(m): return m.description
        case let .declaration(decl): return decl.description
        }
    }
    
}

extension ProgramTree : CustomStringConvertible {

    public var description: String {
        return "(program \(statements.map{$0.description}.joined(separator: " ")))"
    }
    
}

