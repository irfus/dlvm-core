//
//  PrettyPrint.swift
//  LLNM
//
//  Created by Richard Wei on 11/13/16.
//
//

extension Assignment.Value : CustomStringConvertible {
    var description: String {
        switch self {
        case let .log(v): return "log(\(v))"
        case let .sin(v): return "sin(\(v))"
        case let .cos(v): return "cos(\(v))"
        case let .tan(v): return "tan(\(v))"
        case let .exp(v): return "e^\(v)"
        case let .sigmoid(v): return "σ(\(v))"
        case let .relu(v): return "ReLU(\(v))"
        case let .tanh(v): return "tanh(\(v))"
        case let .softmax(v): return "softmax(\(v))"
        case let .negative(v): return "-\(v)"
        case let .add(v1, v2): return "\(v1) + \(v2)"
        case let .sub(v1, v2): return "\(v1) - \(v2)"
        case let .mul(v1, v2): return "\(v1) * \(v2)"
        case let .div(v1, v2): return "\(v1) / \(v2)"
        case let .dot(v1, v2): return "\(v1) • \(v2)"
        }
    }
}

extension Assignment : CustomStringConvertible {
    var description: String {
        return variable + " = " + String(describing: value)
    }
}

extension Graph : CustomStringConvertible {

    public var description: String {
        let tapeDesc = tape.lazy.map{$0.description}.joined(separator: "\n\t")
        return "Expression:\n\t\(root)\nLLNM IR:\n\t\(tapeDesc)"
    }

}

extension Expression : CustomStringConvertible {
    public var description: String {
        switch self {
        case .variable(shape: let shape, name: nil): return "[\(shape)]"
        case let .variable(shape: shape, name: .some(name)): return name + ":[\(shape)]"
        case let .log(expr): return "log(\(expr))"
        case let .sin(expr): return "sin(\(expr))"
        case let .cos(expr): return "cos(\(expr))"
        case let .tan(expr): return "tan(\(expr))"
        case let .exp(expr): return "e^(\(expr))"
        case let .sigmoid(expr): return "σ(\(expr))"
        case let .relu(expr): return "ReLU(\(expr))"
        case let .tanh(expr): return "tanh(\(expr))"
        case let .softmax(expr): return "softmax(\(expr))"
        case let .negative(expr): return "-\(expr)"
        case let .add(lexpr, rexpr): return "\(lexpr) + \(rexpr)"
        case let .sub(lexpr, rexpr): return "\(lexpr) - \(rexpr)"
        case let .mul(lexpr, rexpr): return "\(lexpr) * \(rexpr)"
        case let .div(lexpr, rexpr): return "\(lexpr) / \(rexpr)"
        case let .dot(lexpr, rexpr): return "\(lexpr) • \(rexpr)"
        }
    }
}

extension TensorShape : CustomStringConvertible {
    public var description: String {
        return dimensions.lazy.map{$0.description}.joined(separator: "x")
    }
}
