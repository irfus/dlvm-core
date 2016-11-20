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
    var tape: [Assignment<DataType>] = []
    /// Tensor data
    lazy var data: [Tensor<DataType>] = {
        self.tape.map { assignment in
            switch assignment.rValue {
            case let .input(shape: shape):
                return Tensor(shape: shape, repeating: .zero)
            case let .parameter(shape: shape, initial: initial):
                switch initial {
                case .zeros:
                    return Tensor(shape: assignment.shape, repeating: .zero)
                case let .randomUniform(from: lowerBound, to: upperBound):
                    return Tensor(shape: assignment.shape, factory: {
                        DataType.random(from: lowerBound, to: upperBound)
                    })
                }
            default:
                return Tensor(shape: assignment.shape, repeating: .zero)
            }
        }
    }()
    /// Root of the computation graph
    let root: Expression<DataType>

    /// Initialize from an expression
    /// - parameter expression: neural network expression
    public init(expression: Expression<DataType>) throws {
        root = expression
        try buildIR(from: expression)
    }

    /// Preallocate data for the assignment form
    func preallocateData() {
        _ = data
    }
}

/// RValue of assignment form
indirect enum RValue<T: TensorDataProtocol> {
    case input(shape: TensorShape)
    case parameter(shape: TensorShape, initial: TensorInitializer<T>)
    case log(Assignment<T>)
    case sigmoid(Assignment<T>)
    case relu(Assignment<T>)
    case tanh(Assignment<T>)
    case softmax(Assignment<T>)
    case negative(Assignment<T>)
    case add(Assignment<T>, Assignment<T>)
    case sub(Assignment<T>, Assignment<T>)
    case mul(Assignment<T>, Assignment<T>)
    case div(Assignment<T>, Assignment<T>)
    case product(Assignment<T>, Assignment<T>)
}

/// Assignment in SSA form (phi-function is not currently implemented)
class Assignment<T: TensorDataProtocol> {
    var name: String
    let shape: TensorShape
    let rValue: RValue<T>

    init(name: String, shape: TensorShape, rValue: RValue<T>) {
        self.name = name
        self.shape = shape
        self.rValue = rValue
    }
}

extension RValue : Equatable {
    static func ==<T: TensorDataProtocol>(lhs: RValue<T>, rhs: RValue<T>) -> Bool {
        switch (lhs, rhs) {
        case let (.log(l), .log(r)),
             let (.sigmoid(l), .sigmoid(r)),
             let (.relu(l), .relu(r)),
             let (.tanh(l), .tanh(r)),
             let (.softmax(l), .softmax(r)),
             let (.negative(l), .negative(r)):
            return l == r
        case let (.add(ll, lr), .add(rl, rr)),
             let (.sub(ll, lr), .sub(rl, rr)),
             let (.mul(ll, lr), .mul(rl, rr)),
             let (.div(ll, lr), .div(rl, rr)),
             let (.product(ll, lr), .product(rl, rr)):
            return ll == rl && lr == rr
        default: return false
        }
    }
}

extension Assignment : Equatable {
    static func ==<T: TensorDataProtocol>(lhs: Assignment<T>, rhs: Assignment<T>) -> Bool {
        return lhs === rhs
    }
}

public enum GraphError : Error {
    case productDimensionMismatch(TensorShape, TensorShape)
}

import Foundation

/// Assignment form builder
fileprivate extension Graph {

    /// Build assignment form a neural network expression.
    /// - note: To be called by the initializer.
    /// - parameter expression: neural network expression
    func buildIR(from expression: Expression<DataType>) throws {
        var index: Int = 0

        func newName() -> String {
            index += 1
            return "v\(index)"
        }

        /// Recursively build intermediate representation and store it to the tape
        ///
        /// - Parameter node: top node of the expression
        /// - Returns: top node assignment
        /// - Throws: GraphError: dimension mismatch
        @discardableResult
        func build(_ node: Expression<DataType>) throws -> Assignment<DataType> {
            let assn: Assignment<DataType>
            switch node {
            case let .input(shape: shape, name: name):
                assn = Assignment<DataType>(name: name ?? newName(),
                                            shape: shape,
                                            rValue: .input(shape: shape))

            case let .parameter(shape: shape, initial: initializer, name: name):
                assn = Assignment<DataType>(name: name ?? newName(),
                                            shape: shape,
                                            rValue: .parameter(shape: shape, initial: initializer))

            case let .log(x):
                let op = try build(x)
                assn = Assignment<DataType>(name: newName(),
                                            shape: op.shape,
                                            rValue: .log(op))

            case let .tanh(x):
                let op = try build(x)
                assn = Assignment(name: newName(),
                                  shape: op.shape,
                                  rValue: .tanh(op))

            case let .relu(x):
                let op = try build(x)
                assn = Assignment(name: newName(),
                                  shape: op.shape,
                                  rValue: .relu(op))

            case let .softmax(x):
                let op = try build(x)
                assn = Assignment(name: newName(),
                                  shape: op.shape,
                                  rValue: .softmax(op))

            case let .sigmoid(x):
                let op = try build(x)
                assn = Assignment(name: newName(),
                                  shape: op.shape,
                                  rValue: .sigmoid(op))

            case let .negative(x):
                let op = try build(x)
                assn = Assignment(name: newName(),
                                  shape: op.shape,
                                  rValue: .sigmoid(op))

            case let .add(lhs, rhs):
                let leftOp = try build(lhs), rightOp = try build(rhs)
                assn = Assignment(name: newName(),
                                  shape: leftOp.shape,
                                  rValue: .add(leftOp, rightOp))


            case let .sub(lhs, rhs):
                let leftOp = try build(lhs), rightOp = try build(rhs)
                assn = Assignment(name: newName(),
                                  shape: leftOp.shape,
                                  rValue: .sub(leftOp, rightOp))

            case let .mul(lhs, rhs):
                let leftOp = try build(lhs), rightOp = try build(rhs)
                assn = Assignment(name: newName(),
                                  shape: leftOp.shape,
                                  rValue: .mul(leftOp, rightOp))

            case let .div(lhs, rhs):
                let leftOp = try build(lhs), rightOp = try build(rhs)
                assn = Assignment(name: newName(),
                                  shape: leftOp.shape,
                                  rValue: .div(leftOp, rightOp))

            case let .product(lhs, rhs):
                let leftOp = try build(lhs), rightOp = try build(rhs)
                let leftDim = leftOp.shape.dimensions
                let rightDim = rightOp.shape.dimensions
                guard leftDim.last == rightDim.first else {
                    throw GraphError.productDimensionMismatch(leftOp.shape, rightOp.shape)
                }
                let newDim = leftDim.dropLast() + rightDim.dropFirst()
                let newShape = TensorShape(newDim)
                assn = Assignment(name: newName(),
                                  shape: newShape,
                                  rValue: .product(leftOp, rightOp))

            case let .layer(subExpr, name: name):
                let op = try build(subExpr)
                op.name = name
                return op
            }

            /// Add assignment to the tape
            tape.append(assn)
            return assn
        }
        
        try build(expression)
    }
    
}
