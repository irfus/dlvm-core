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

public indirect enum Node<DataType: TensorDataProtocol> {
    case variable(Tensor<DataType>)
    case log(Node)
    case sin(Node)
    case cos(Node)
    case tan(Node)
    case exp(Node)
    case sigmoid(Node)
    case relu(Node)
    case tanh(Node)
    case negative(Node)
    case plus(Node, Node)
    case minus(Node, Node)
    case times(Node, Node)
    case divide(Node, Node)
    case dot(Node, Node)
}

extension Node : Chainable {

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

public extension Node {

    public static func +(lhs: Node<DataType>, rhs: Node<DataType>) -> Node {
        return .plus(lhs, rhs)
    }

    public static func -(lhs: Node<DataType>, rhs: Node<DataType>) -> Node {
        return .minus(lhs, rhs)
    }

    public static func *(lhs: Node<DataType>, rhs: Node<DataType>) -> Node {
        return .times(lhs, rhs)
    }

    public static func /(lhs: Node<DataType>, rhs: Node<DataType>) -> Node {
        return .divide(lhs, rhs)
    }

    public static func •(lhs: Node<DataType>, rhs: Node<DataType>) -> Node {
        return .dot(lhs, rhs)
    }

}
