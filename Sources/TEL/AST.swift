//
//  AST.swift
//  DLVM
//
//  Copyright 2016-2017 Richard Wei.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

import struct Parsey.SourceRange

public protocol ASTNode {
    var range: SourceRange { get }
}

public enum Attribute : ASTNode {
    case name(String, SourceRange)
    case type(String, SourceRange)

    public var range: SourceRange {
        switch self {
        case let .name(_, sr), let .type(_, sr):
            return sr
        }
    }
}

public enum Variable : ASTNode {
    case simple(String, SourceRange)
    case recurrent(String, timeStep: String, offset: Int, SourceRange)

    public var name: String {
        switch self {
        case let .simple(name, _), let .recurrent(name, _, _, _):
            return name
        }
    }

    public var range: SourceRange {
        switch self {
        case .simple(_, let sr), .recurrent(_, timeStep: _, offset: _, let sr):
            return sr
        }
    }
}

public enum Role {
    case input, output, hidden, parameter
}

public enum Constant : ASTNode {
    case int(Int, SourceRange)
    case float(Double, SourceRange)

    public var range: SourceRange {
        switch self {
        case .int(_, let sr), .float(_, let sr):
            return sr
        }
    }
}

public indirect enum Expression : ASTNode {
    public enum InfixOperator {
        case add, sub, mul, div
    }
    /// Integer
    case constant(Constant, SourceRange)
    /// Random
    case random(Constant, Constant, SourceRange)
    /// Variable
    case variable(Variable, SourceRange)
    /// Intrinsic call
    case call(String, [Expression], SourceRange)
    /// Negation
    case negate(Expression, SourceRange)
    /// Element-wise addition
    case infixOp(InfixOperator, Expression, Expression, SourceRange)
    /// Tensor product
    case product(Expression, Expression, SourceRange)
    /// Concatenation
    case concat([Expression], dimension: Int, SourceRange)
    /// Reshape
    case reshape(Expression, shape: [Int], SourceRange)
    /// Transpose
    case transpose(Expression, SourceRange)

    public var range: SourceRange {
        switch self {
            case .constant(_, let sr),
                 .random(_, _, let sr),
                 .variable(_, let sr),
                 .call(_, _, let sr),
                 .negate(_, let sr),
                 .infixOp(_, _, _, let sr),
                 .product(_, _, let sr),
                 .concat(_, _, let sr),
                 .reshape(_, _, let sr),
                 .transpose(_, let sr):
            return sr
        }
    }
}

public indirect enum Declaration : ASTNode {
    case assignment(Variable, Role, [Int], Expression?, SourceRange)
    case recurrence(String, [Declaration], SourceRange)

    public var range: SourceRange {
        switch self {
        case .assignment(_, _, _, _, let sr),
             .recurrence(_, _, let sr):
            return sr
        }
    }
}

public enum Statement : ASTNode {
    case attribute(Attribute, SourceRange)
    case declaration(Declaration, SourceRange)

    public var range: SourceRange {
        switch self {
        case .attribute(_, let sr),
             .declaration(_, let sr):
            return sr
        }
    }
}

public struct ProgramTree : ASTNode {
    public let statements: [Statement]
    public let range: SourceRange
}

extension Constant : CustomStringConvertible {
    public var description: String {
        switch self {
        case let .int(i, _):   return i.description
        case let .float(f, _): return f.description
        }
    }
}

extension Variable : CustomStringConvertible {

    public var description: String {
        switch self {
        case let .simple(name, _): return name
        case let .recurrent(name, timeStep: t, offset: offset, _):
            return name + "[" + t + offset.description + "]"
        }
    }
    
}

extension Expression.InfixOperator : CustomStringConvertible {
    public var description: String {
        switch self {
        case .add: return "+"
        case .sub: return "-"
        case .mul: return "*"
        case .div: return "/"
        }
    }
}

extension Expression : CustomStringConvertible {
    public var description: String {
        switch self {
        case let .constant(c, _): return c.description
        case let .random(lhs, rhs, _): return "(random \(lhs) \(rhs))"
        case let .variable(v, _): return v.name
        case let .call(name, args, _):
            return "(\(name) \(args.map{$0.description}.joined(separator: " ")))"
        case let .negate(expr, _):
            return "(- \(expr))"
        case let .infixOp(op, lhs, rhs, _):
            return "(\(op) \(lhs) \(rhs))"
        case let .product(lhs, rhs, _):
            return "(⊗ \(lhs) \(rhs))"
        case let .concat(exprs, dim, _):
            return "(concat \(dim) \(exprs.map{$0.description}.joined(separator: " ")))"
        case let .reshape(expr, shape: shape, _):
            return "(reshape \(expr) [\(shape.map{$0.description}.joined(separator: "x"))])"
        case let .transpose(expr, _):
            return "(transpose \(expr))"
        }
    }
}

extension Declaration : CustomStringConvertible {
    public var description: String {
        switch self {
        case let .assignment(v, role, dim, expr, _):
            return "(define \(v):\(role)[\(dim.map{$0.description}.joined(separator: "x"))]\(expr == nil ? "" : " " + expr!.description))"
        case let .recurrence(timestep, decls, _):
            return "(recurrent \(timestep) \(decls.map{$0.description}.joined(separator: " ")))"
        }
    }
}

extension Attribute: CustomStringConvertible {

    public var description: String {
        switch self {
        case let .name(n, _): return "(define-module-name \(n))"
        case let .type(t, _): return "(define-type \(t))"
        }
    }

}

extension Statement : CustomStringConvertible {

    public var description: String {
        switch self {
        case let .attribute(m, _): return m.description
        case let .declaration(decl, _): return decl.description
        }
    }
    
}

extension ProgramTree : CustomStringConvertible {

    public var description: String {
        return "(program \(statements.map{$0.description}.joined(separator: " ")))"
    }
    
}

