//
//  Expression.swift
//  LLNM
//
//  Created by Richard Wei on 11/6/16.
//
//

/// Symbolic expression of neural network
public indirect enum Expression<DataType: TensorDataProtocol> {
    /// Input tensor placeholder
    case input(shape: TensorShape, name: String?)
    /// Parameter of the network
    case parameter(shape: TensorShape, initial: TensorInitializer<DataType>, name: String?)
    /// Logarithm
    case log(Expression)
    /// Sigmoid
    case sigmoid(Expression)
    /// ReLU
    case relu(Expression)
    /// Tanh
    case tanh(Expression)
    /// Softmax
    case softmax(Expression)
    /// Negation
    case negative(Expression)
    /// Element-wise addition
    case add(Expression, Expression)
    /// Element-wise subtraction
    case sub(Expression, Expression)
    /// Element-wise product
    case mul(Expression, Expression)
    /// Element-wise quotient
    case div(Expression, Expression)
    /// Tensor product
    case product(Expression, Expression)
    /// Named layer
    case layer(Expression, name: String)
}

infix operator • : MultiplicationPrecedence
infix operator <-

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
        return .product(lhs, rhs)
    }

    @inline(__always)
    public static prefix func -(rhs: Expression) -> Expression {
        return .negative(rhs)
    }

}

/// Free functions that assist expression building in mathematical sense
@inline(__always)
public func sigmoid<T: TensorDataProtocol>(_ expression: Expression<T>) -> Expression<T> {
    return .sigmoid(expression)
}

@inline(__always)
public func relu<T: TensorDataProtocol>(_ expression: Expression<T>) -> Expression<T> {
    return .relu(expression)
}

@inline(__always)
public func tanh<T: TensorDataProtocol>(_ expression: Expression<T>) -> Expression<T> {
    return .tanh(expression)
}

@inline(__always)
public func softmax<T: TensorDataProtocol>(_ expression: Expression<T>) -> Expression<T> {
    return .softmax(expression)
}

@inline(__always)
public func <-<T: TensorDataProtocol>(lhs: Expression<T>, rhs: String) -> Expression<T> {
    return .layer(lhs, name: rhs)
}
