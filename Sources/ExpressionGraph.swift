//
//  ExpressionGraph.swift
//  LLNM
//
//  Created by Richard Wei on 11/11/16.
//
//

/// Assignment in SSA form
struct Assignment {
    typealias Variable = String
    
    enum Value {
        case variable(Variable)
        case log(Variable)
        case sin(Variable)
        case cos(Variable)
        case tan(Variable)
        case exp(Variable)
        case sigmoid(Variable)
        case relu(Variable)
        case tanh(Variable)
        case negative(Variable)
        case add(Variable, Variable)
        case sub(Variable, Variable)
        case mul(Variable, Variable)
        case div(Variable, Variable)
        case matMul(Variable, Variable)
    }
    let variable: Variable
    let value: Value
}

extension Assignment.Value : Equatable {
    static func ==(lhs: Assignment.Value, rhs: Assignment.Value) -> Bool {
        switch (lhs, rhs) {
        case let (.variable(l), .variable(r)),
             let (.log(l), .log(r)),
             let (.sin(l), .sin(r)),
             let (.cos(l), .cos(r)),
             let (.tan(l), .tan(r)),
             let (.exp(l), .exp(r)),
             let (.sigmoid(l), .sigmoid(r)),
             let (.relu(l), .relu(r)),
             let (.tanh(l), .tanh(r)),
             let (.negative(l), .negative(r)):
            return l == r
        case let (add(ll, lr), add(rl, rr)),
             let (sub(ll, lr), sub(rl, rr)),
             let (mul(ll, lr), mul(rl, rr)),
             let (div(ll, lr), div(rl, rr)),
             let (matMul(ll, lr), matMul(rl, rr)):
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

public class ExpressionGraph<DataType : TensorDataProtocol> {

    var tape: [Assignment] = []
    var shapes: [TensorShape] = []
    let root: Expression<DataType>

    public init(expression: Expression<DataType>) {
        self.root = expression
        buildAssignmentForm(from: expression)
    }

}

infix operator <-
fileprivate func <-(lhs: Assignment.Variable, rhs: Assignment.Value) -> Assignment {
    return Assignment(variable: lhs, value: rhs)
}

extension ExpressionGraph {

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
            case let .tensor(shape, name: name):
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
                tape.append(retVar <- .exp(op))

            case let .relu(expr):
                let op = build(expr)
                retVar = newVar()
                tape.append(retVar <- .exp(op))

            case let .tanh(expr):
                let op = build(expr)
                retVar = newVar()
                tape.append(retVar <- .tanh(op))

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

            case let .matMul(lhs, rhs):
                let lop = build(lhs), rop = build(rhs)
                retVar = newVar()
                tape.append(retVar <- .matMul(lop, rop))
            }
            return retVar
        }

        build(expression)
    }

}
