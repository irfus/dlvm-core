//
//  PrettyPrint.swift
//  LLNM
//
//  Created by Richard Wei on 11/13/16.
//
//

// This file contains string conversion methods for pretty printing

extension RValue : CustomStringConvertible {
    var description: String {
        switch self {
        case let .input(shape: shape): return "input:[\(shape)]"
        case let .parameter(shape: shape, initial: _): return "param:[\(shape)]"
        case let .log(v): return "log(\(v.name))"
        case let .sigmoid(v): return "σ(\(v.name))"
        case let .relu(v): return "ReLU(\(v.name))"
        case let .tanh(v): return "tanh(\(v.name))"
        case let .softmax(v): return "softmax(\(v.name))"
        case let .negative(v): return "-\(v.name)"
        case let .add(v1, v2): return "\(v1.name) + \(v2.name)"
        case let .mul(v1, v2): return "\(v1.name) * \(v2.name)"
        case let .min(v1, v2): return "\(v1.name) - \(v2.name)"
        case let .max(v1, v2): return "\(v1.name) / \(v2.name)"
        case let .product(v1, v2): return "\(v1.name) • \(v2.name)"
        case let .scalarComplement(v1, v2): return "\(v1) - \(v2.name)"
        }
    }
}

extension Variable : CustomStringConvertible {
    var description: String {
        return "[\(shape)] \(name) = \(rValue)"
    }
}

extension Graph : CustomStringConvertible {
    public var description: String {
        let tapeDesc = tape.lazy.map{$0.description}.joined(separator: "\n\t")
        return "Expression:\n\t\(root)\nIR:\n\t\(tapeDesc)"
    }
}

extension Expression : CustomStringConvertible {
    public var description: String {
        switch self {
        case let .input(shape: shape, name: nil):
            return "[\(shape)]"
        case let .input(shape: shape, name: name?):
            return name + ":[\(shape)]"
        case let .parameter(shape: shape, initial: _, name: nil):
            return "[\(shape)]"
        case let .parameter(shape: shape, initial: _, name: name?):
            return name + ":[\(shape)]"
        case let .log(expr): return "log(\(expr))"
        case let .sigmoid(expr): return "σ(\(expr))"
        case let .relu(expr): return "ReLU(\(expr))"
        case let .tanh(expr): return "tanh(\(expr))"
        case let .softmax(expr): return "softmax(\(expr))"
        case let .negative(expr): return "-\(expr)"
        case let .add(lexpr, rexpr): return "\(lexpr) + \(rexpr)"
        case let .mul(lexpr, rexpr): return "\(lexpr) * \(rexpr)"
        case let .min(lexpr, rexpr): return "\(lexpr) - \(rexpr)"
        case let .max(lexpr, rexpr): return "\(lexpr) / \(rexpr)"
        case let .product(lexpr, rexpr): return "\(lexpr) • \(rexpr)"
        case let .layer(expr, name: name): return "\(name)=(\(expr))"
        case let .scalarComplement(scalar, expr): return "\(scalar) - \(expr)"
        }
    }
}

extension TensorShape : CustomStringConvertible {
    public var description: String {
        return dimensions.lazy.map{$0.description}.joined(separator: "x")
    }
}

extension GraphError : CustomStringConvertible {
    public var description: String {
        switch self {
        case let .productDimensionMismatch(lShape, rShape):
            return "Dimension mismatch: \(lShape) cannot form a product with \(rShape)"
        }
    }
}
