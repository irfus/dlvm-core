//
//  Graph.swift
//  LLNM
//
//  Created by Richard Wei on 11/11/16.
//
//

/// This file contains LLNM Intermediate Representation (computation graph).
/// We will use the acronym **LLNM IR**.

/// Computation graph of the neural network
/// - parameter DataType: type of elements of the tensor (Float, Double, ...)
public class Graph<DataType : TensorDataProtocol> {
    /// Computation tape (SSA form)
    var tape: [Assignment] = []
    /// Tensor shapes for variables in the computation tape
    var shapes: [TensorShape] = []
    /// Tensors
    lazy var tensors: [Tensor<DataType>] = {
        self.shapes.map { Tensor(shape: $0) }
    }()
    /// Root of the computation graph
    let root: Expression

    /// Initialize from an expression
    /// - parameter expression: Neural network expression
    public init(expression: Expression) {
        root = expression
        buildIR(from: expression)
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
/// Shortcut constructor for Assignment
@inline(__always)
fileprivate func <-(lhs: Assignment.Variable, rhs: Assignment.Value) -> Assignment {
    return Assignment(variable: lhs, value: rhs)
}

/// Assignment form builder
fileprivate extension Graph {

    /// Build assignment form a neural network expression.
    /// - note: To be called by the initializer.
    /// - parameter expression: neural network expression
    func buildIR(from expression: Expression) {
        var index: Int = 0

        func newVar() -> Assignment.Variable {
            index += 1
            return "%v\(index)"
        }

        @discardableResult
        func build(_ expression: Expression) -> (Assignment.Variable, TensorShape) {
            let newVariable: Assignment.Variable
            let newShape: TensorShape
            let assignment: Assignment
            switch expression {
            case let .variable(shape, name: name):
                return (name ?? newVar(), shape)

            case let .log(expr):
                let (op, shape) = build(expr)
                newVariable = newVar()
                newShape = shape
                assignment = newVariable <- .log(op)

            case let .sin(expr):
                let (op, shape) = build(expr)
                newVariable = newVar()
                newShape = shape
                assignment = newVariable <- .sin(op)

            case let .cos(expr):
                let (op, shape) = build(expr)
                newVariable = newVar()
                newShape = shape
                assignment = newVariable <- .cos(op)

            case let .tan(expr):
                let (op, shape) = build(expr)
                newVariable = newVar()
                newShape = shape
                assignment = newVariable <- .tan(op)

            case let .exp(expr):
                let (op, shape) = build(expr)
                newVariable = newVar()
                newShape = shape
                assignment = newVariable <- .exp(op)

            case let .sigmoid(expr):
                let (op, shape) = build(expr)
                newVariable = newVar()
                newShape = shape
                assignment = newVariable <- .sigmoid(op)

            case let .relu(expr):
                let (op, shape) = build(expr)
                newVariable = newVar()
                newShape = shape
                assignment = newVariable <- .relu(op)

            case let .tanh(expr):
                let (op, shape) = build(expr)
                newVariable = newVar()
                newShape = shape
                assignment = newVariable <- .tanh(op)

            case let .softmax(expr):
                let (op, shape) = build(expr)
                newVariable = newVar()
                newShape = shape
                assignment = newVariable <- .softmax(op)

            case let .negative(expr):
                let (op, shape) = build(expr)
                newVariable = newVar()
                newShape = shape
                assignment = newVariable <- .negative(op)

            case let .add(lhs, rhs):
                let (lop, lshape) = build(lhs), (rop, _) = build(rhs)
                newVariable = newVar()
                newShape = lshape
                assignment = newVariable <- .add(lop, rop)

            case let .sub(lhs, rhs):
                let (lop, lshape) = build(lhs), (rop, _) = build(rhs)
                newVariable = newVar()
                newShape = lshape
                assignment = newVariable <- .sub(lop, rop)

            case let .mul(lhs, rhs):
                let (lop, lshape) = build(lhs), (rop, _) = build(rhs)
                newVariable = newVar()
                newShape = lshape
                assignment = newVariable <- .mul(lop, rop)

            case let .div(lhs, rhs):
                let (lop, lshape) = build(lhs), (rop, _) = build(rhs)
                newVariable = newVar()
                newShape = lshape
                assignment = newVariable <- .div(lop, rop)

            case let .dot(lhs, rhs):
                let (lop, lshape) = build(lhs), (rop, rshape) = build(rhs)
                newVariable = newVar()
                // TODO: Shape calculation
                newShape = lshape
                assignment = newVariable <- .dot(lop, rop)
                
            case let .layer(expr, name: name):
                return build(expr)

            }
            tape.append(assignment)
            shapes.append(newShape)
            return (newVariable, newShape)
        }

        build(expression)
    }

}
