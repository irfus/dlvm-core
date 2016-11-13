//
//  ExpressionGraph.swift
//  LLNM
//
//  Created by Richard Wei on 11/11/16.
//
//

/// This file contains LLNM Intermediate Representation (computation graph).

/// Computation graph of the neural network
/// - parameter DataType: type of elements of the tensor (Float, Double, ...)
public class Graph<DataType : TensorDataProtocol> {
    /// Computation tape (SSA form)
    var tape: [Assignment] = []
    /// Tensor shapes for variables in the computation tape
    var shapes: [TensorShape] = []
    /// Root of the computation graph
    let root: Expression<DataType>

    /// Initialize from an expression
    /// - parameter expression: Neural network expression
    public init(expression: Expression<DataType>) {
        root = expression
        buildAssignmentForm(from: expression)
    }
}

/// Assignment in SSA form
struct Assignment {
    typealias Variable = String
    enum Value {
        case log(Variable)
        case sin(Variable)
        case cos(Variable)
        case tan(Variable)
        case exp(Variable)
        case sigmoid(Variable)
        case relu(Variable)
        case tanh(Variable)
        case softmax(Variable)
        case negative(Variable)
        case add(Variable, Variable)
        case sub(Variable, Variable)
        case mul(Variable, Variable)
        case div(Variable, Variable)
        case dot(Variable, Variable)
    }
    let variable: Variable
    let value: Value
}

extension Assignment.Value : Equatable {
    static func ==(lhs: Assignment.Value, rhs: Assignment.Value) -> Bool {
        switch (lhs, rhs) {
        case let (.log(l), .log(r)),
             let (.sin(l), .sin(r)),
             let (.cos(l), .cos(r)),
             let (.tan(l), .tan(r)),
             let (.exp(l), .exp(r)),
             let (.sigmoid(l), .sigmoid(r)),
             let (.relu(l), .relu(r)),
             let (.tanh(l), .tanh(r)),
             let (.softmax(l), .softmax(r)),
             let (.negative(l), .negative(r)):
            return l == r
        case let (add(ll, lr), add(rl, rr)),
             let (sub(ll, lr), sub(rl, rr)),
             let (mul(ll, lr), mul(rl, rr)),
             let (div(ll, lr), div(rl, rr)),
             let (dot(ll, lr), dot(rl, rr)):
            return ll == rl && lr == rr
        default: return false
        }
    }
}

extension Assignment : Equatable {
    static func ==(lhs: Assignment, rhs: Assignment) -> Bool {
        return lhs.value == rhs.value && lhs.variable == rhs.variable
    }
}

infix operator <-
fileprivate func <-(lhs: Assignment.Variable, rhs: Assignment.Value) -> Assignment {
    return Assignment(variable: lhs, value: rhs)
}

/// Assignment form builder
fileprivate extension Graph {

    /// Build assignment form a neural network expression.
    /// - note: To be called by the initializer.
    /// - parameter expression: neural network expression
    func buildAssignmentForm(from expression: Expression<DataType>) {
        var index: Int = 0

        func newVar() -> Assignment.Variable {
            index += 1
            return "v\(index)"
        }

        @discardableResult
        func build(_ expression: Expression<DataType>) -> Assignment.Variable {
            let retVar: Assignment.Variable
            switch expression {
            case let .variable(shape, name: name):
                return name ?? newVar()

            case let .log(expr):
                let op = build(expr)
                retVar = newVar()
                tape.append(retVar <- .log(op))

            case let .sin(expr):
                let op = build(expr)
                retVar = newVar()
                tape.append(retVar <- .sin(op))

            case let .cos(expr):
                let op = build(expr)
                retVar = newVar()
                tape.append(retVar <- .cos(op))

            case let .tan(expr):
                let op = build(expr)
                retVar = newVar()
                tape.append(retVar <- .tan(op))

            case let .exp(expr):
                let op = build(expr)
                retVar = newVar()
                tape.append(retVar <- .exp(op))

            case let .sigmoid(expr):
                let op = build(expr)
                retVar = newVar()
                tape.append(retVar <- .sigmoid(op))

            case let .relu(expr):
                let op = build(expr)
                retVar = newVar()
                tape.append(retVar <- .relu(op))

            case let .tanh(expr):
                let op = build(expr)
                retVar = newVar()
                tape.append(retVar <- .tanh(op))

            case let .softmax(expr):
                let op = build(expr)
                retVar = newVar()
                tape.append(retVar <- .softmax(op))

            case let .negative(expr):
                let op = build(expr)
                retVar = newVar()
                tape.append(retVar <- .negative(op))

            case let .add(lhs, rhs):
                let lop = build(lhs), rop = build(rhs)
                retVar = newVar()
                tape.append(retVar <- .add(lop, rop))

            case let .sub(lhs, rhs):
                let lop = build(lhs), rop = build(rhs)
                retVar = newVar()
                tape.append(retVar <- .sub(lop, rop))

            case let .mul(lhs, rhs):
                let lop = build(lhs), rop = build(rhs)
                retVar = newVar()
                tape.append(retVar <- .mul(lop, rop))

            case let .div(lhs, rhs):
                let lop = build(lhs), rop = build(rhs)
                retVar = newVar()
                tape.append(retVar <- .div(lop, rop))

            case let .dot(lhs, rhs):
                let lop = build(lhs), rop = build(rhs)
                retVar = newVar()
                tape.append(retVar <- .dot(lop, rop))
            }
            return retVar
        }

        build(expression)
    }

}

extension Assignment : CustomStringConvertible {
    var description: String {
        return variable + " <- " + String(describing: value)
    }
}

extension Graph : CustomStringConvertible {

    public var description: String {
        let tapeDesc = tape.lazy.map{$0.description}.joined(separator: "\n\t")
        return "Expression:\t\(root)\nComputation Tape:\n\t\(tapeDesc)"
    }

}
