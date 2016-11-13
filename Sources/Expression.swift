//
//  Expression.swift
//  LLNM
//
//  Created by Richard Wei on 11/6/16.
//
//

import CCuDNN

public protocol Chainable {
    associatedtype DataType
    mutating func forward()
    mutating func backward()
    func evaluated() -> DataType
}

public indirect enum Expression<DataType : TensorDataProtocol> {
    case tensor(shape: TensorShape, name: String?)
    case log(Expression)
    case sin(Expression)
    case cos(Expression)
    case tan(Expression)
    case exp(Expression)
    case sigmoid(Expression)
    case relu(Expression)
    case tanh(Expression)
    case negative(Expression)
    case add(Expression, Expression)
    case sub(Expression, Expression)
    case mul(Expression, Expression)
    case div(Expression, Expression)
    case matMul(Expression, Expression)
}

extension Expression : Chainable {

    public mutating func forward() {
        
    }

    public mutating func backward() {
        
    }

    public func evaluated() -> DataType {
        let ret: DataType! = nil
        return ret
    }

}

infix operator • : MultiplicationPrecedence

public extension Expression {

    public static func +(lhs: Expression<DataType>, rhs: Expression<DataType>) -> Expression {
        return .add(lhs, rhs)
    }

    public static func -(lhs: Expression<DataType>, rhs: Expression<DataType>) -> Expression {
        return .sub(lhs, rhs)
    }

    public static func *(lhs: Expression<DataType>, rhs: Expression<DataType>) -> Expression {
        return .mul(lhs, rhs)
    }

    public static func /(lhs: Expression<DataType>, rhs: Expression<DataType>) -> Expression {
        return .div(lhs, rhs)
    }

    public static func •(lhs: Expression<DataType>, rhs: Expression<DataType>) -> Expression {
        return .matMul(lhs, rhs)
    }

    public static prefix func -(rhs: Expression<DataType>) -> Expression {
        return .negative(rhs)
    }

}

/// Free functions that assist expression building in a mathematical sense

@inline(__always)
public func log<T: TensorDataProtocol>(_ expression: Expression<T>) -> Expression<T> {
    return .log(expression)
}

@inline(__always)
public func sin<T: TensorDataProtocol>(_ expression: Expression<T>) -> Expression<T> {
    return .sin(expression)
}

@inline(__always)
public func cos<T: TensorDataProtocol>(_ expression: Expression<T>) -> Expression<T> {
    return .cos(expression)
}

@inline(__always)
public func tan<T: TensorDataProtocol>(_ expression: Expression<T>) -> Expression<T> {
    return .tan(expression)
}

@inline(__always)
public func exp<T: TensorDataProtocol>(_ expression: Expression<T>) -> Expression<T> {
    return .exp(expression)
}
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
