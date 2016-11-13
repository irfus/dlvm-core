//
//  Expression.swift
//  LLNM
//
//  Created by Richard Wei on 11/6/16.
//
//

import CCuDNN

/// Symbolic expression of neural network
/// - parameter DataType: type of elements of the tensor (Float, Double, ...)
public indirect enum Expression {
    case variable(shape: TensorShape, name: String?)
    case log(Expression)
    case sin(Expression)
    case cos(Expression)
    case tan(Expression)
    case exp(Expression)
    case sigmoid(Expression)
    case relu(Expression)
    case tanh(Expression)
    case softmax(Expression)
    case negative(Expression)
    case add(Expression, Expression)
    case sub(Expression, Expression)
    case mul(Expression, Expression)
    case div(Expression, Expression)
    case dot(Expression, Expression)
}

infix operator • : MultiplicationPrecedence

public extension Expression {

    @inline(__always)
    public static func +(lhs: Expression, rhs: Expression) -> Expression {
        return .add(lhs, rhs)
    }

    @inline(__always)
    public static func -(lhs: Expression, rhs: Expression) -> Expression {
        return .sub(lhs, rhs)
    }

    @inline(__always)
    public static func *(lhs: Expression, rhs: Expression) -> Expression {
        return .mul(lhs, rhs)
    }

    @inline(__always)
    public static func /(lhs: Expression, rhs: Expression) -> Expression {
        return .div(lhs, rhs)
    }

    @inline(__always)
    public static func •(lhs: Expression, rhs: Expression) -> Expression {
        return .dot(lhs, rhs)
    }

    @inline(__always)
    public static prefix func -(rhs: Expression) -> Expression {
        return .negative(rhs)
    }

}

/// Free functions that assist expression building in mathematical sense

@inline(__always)
public func log(_ expression: Expression) -> Expression {
    return .log(expression)
}

@inline(__always)
public func sin(_ expression: Expression) -> Expression {
    return .sin(expression)
}

@inline(__always)
public func cos(_ expression: Expression) -> Expression {
    return .cos(expression)
}

@inline(__always)
public func tan(_ expression: Expression) -> Expression {
    return .tan(expression)
}

@inline(__always)
public func exp(_ expression: Expression) -> Expression {
    return .exp(expression)
}
@inline(__always)
public func sigmoid(_ expression: Expression) -> Expression {
    return .sigmoid(expression)
}

@inline(__always)
public func relu(_ expression: Expression) -> Expression {
    return .relu(expression)
}

@inline(__always)
public func tanh(_ expression: Expression) -> Expression {
    return .tanh(expression)
}

@inline(__always)
public func softmax(_ expression: Expression) -> Expression {
    return .softmax(expression)
}
