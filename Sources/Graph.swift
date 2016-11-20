//
//  Graph.swift
//  LLNM
//
//  Created by Richard Wei on 11/11/16.
//
//

/// This file contains LLNM Intermediate Representation (computation graph).
/// We will use the acronym **LLNM IR**.

import CUDARuntime

/// Computation graph of the neural network
/// - parameter DataType: type of elements of the tensor (Float, Double, ...)
public class Graph<DataType : TensorDataProtocol> {
    /// Computation tape (SSA form)
    var tape: [Assignment<DataType>] = []

    /// Root of the computation graph
    let root: Expression<DataType>

    /// cuDNN instance
    let dnn: DNN

    /// op tensor
    lazy var tensorOperators: TensorOperators<DataType> = { TensorOperators() }()

    /// Initialize from an expression
    /// - parameter expression: neural network expression
    public init(expression: Expression<DataType>, device: Device = Device.current) throws {
        root = expression
        self.dnn = DNN.shared(on: device)
        try buildIR(from: expression)
    }

    /// Preallocate data for the assignment form
    func preallocateData() {
        tape.forEach { _ = $0.data }
    }
}

/// RValue of assignment form
indirect enum RValue<DataType: TensorDataProtocol> {
    case input(shape: TensorShape)
    case parameter(shape: TensorShape, initial: TensorInitializer<DataType>)
    case log(Assignment<DataType>)
    case sigmoid(Assignment<DataType>)
    case relu(Assignment<DataType>)
    case tanh(Assignment<DataType>)
    case softmax(Assignment<DataType>)
    case negative(Assignment<DataType>)
    case add(Assignment<DataType>, Assignment<DataType>)
    case mul(Assignment<DataType>, Assignment<DataType>)
    case min(Assignment<DataType>, Assignment<DataType>)
    case max(Assignment<DataType>, Assignment<DataType>)
    case product(Assignment<DataType>, Assignment<DataType>)
    case scalarComplement(DataType, Assignment<DataType>)
}

/// Assignment in SSA form (phi-function is not currently implemented)
class Assignment<DataType: TensorDataProtocol> {
    var name: String
    let shape: TensorShape
    let rValue: RValue<DataType>
    unowned let graph: Graph<DataType>

    lazy var data: Tensor<DataType> = {
        switch self.rValue {
        case let .input(shape: shape):
            return Tensor(shape: shape, repeating: .zero)
        case let .parameter(shape: shape, initial: initial):
            switch initial {
            case .zeros:
                return Tensor(shape: self.shape, repeating: .zero)
            case let .randomUniform(from: lowerBound, to: upperBound):
                return Tensor(shape: self.shape, factory: {
                    DataType.random(from: lowerBound, to: upperBound)
                })
            }
        default:
            return Tensor(shape: self.shape, repeating: .zero)
        }
    }()

    init(name: String, shape: TensorShape, rValue: RValue<DataType>, graph: Graph<DataType>) {
        self.name = name
        self.shape = shape
        self.rValue = rValue
        self.graph = graph
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
             let (.mul(ll, lr), .mul(rl, rr)),
             let (.min(ll, lr), .min(rl, rr)),
             let (.max(ll, lr), .max(rl, rr)),
             let (.product(ll, lr), .product(rl, rr)):
            return ll == rl && lr == rr
        case let (.scalarComplement(ll, lr), .scalarComplement(rl, rr)):
            return ll == rl && lr == rr
        default: return false
        }
    }
}

extension Assignment : Equatable {
    static func ==(lhs: Assignment<DataType>, rhs: Assignment<DataType>) -> Bool {
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
                                            rValue: .input(shape: shape),
                                            graph: self)

            case let .parameter(shape: shape, initial: initializer, name: name):
                assn = Assignment<DataType>(name: name ?? newName(),
                                            shape: shape,
                                            rValue: .parameter(shape: shape, initial: initializer),
                                            graph: self)

            case let .log(x):
                let op = try build(x)
                assn = Assignment<DataType>(name: newName(),
                                            shape: op.shape,
                                            rValue: .log(op),
                                            graph: self)

            case let .tanh(x):
                let op = try build(x)
                assn = Assignment(name: newName(),
                                  shape: op.shape,
                                  rValue: .tanh(op),
                                  graph: self)

            case let .relu(x):
                let op = try build(x)
                assn = Assignment(name: newName(),
                                  shape: op.shape,
                                  rValue: .relu(op),
                                  graph: self)

            case let .softmax(x):
                let op = try build(x)
                assn = Assignment(name: newName(),
                                  shape: op.shape,
                                  rValue: .softmax(op),
                                  graph: self)

            case let .sigmoid(x):
                let op = try build(x)
                assn = Assignment(name: newName(),
                                  shape: op.shape,
                                  rValue: .sigmoid(op),
                                  graph: self)

            case let .negative(x):
                let op = try build(x)
                assn = Assignment(name: newName(),
                                  shape: op.shape,
                                  rValue: .sigmoid(op),
                                  graph: self)

            case let .add(lhs, rhs):
                let leftOp = try build(lhs), rightOp = try build(rhs)
                assn = Assignment(name: newName(),
                                  shape: leftOp.shape,
                                  rValue: .add(leftOp, rightOp),
                                  graph: self)


            case let .mul(lhs, rhs):
                let leftOp = try build(lhs), rightOp = try build(rhs)
                assn = Assignment(name: newName(),
                                  shape: leftOp.shape,
                                  rValue: .mul(leftOp, rightOp),
                                  graph: self)

            case let .min(lhs, rhs):
                let leftOp = try build(lhs), rightOp = try build(rhs)
                assn = Assignment(name: newName(),
                                  shape: leftOp.shape,
                                  rValue: .min(leftOp, rightOp),
                                  graph: self)

            case let .max(lhs, rhs):
                let leftOp = try build(lhs), rightOp = try build(rhs)
                assn = Assignment(name: newName(),
                                  shape: leftOp.shape,
                                  rValue: .max(leftOp, rightOp),
                                  graph: self)

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
                                  rValue: .product(leftOp, rightOp),
                                  graph: self)

            case let .layer(subExpr, name: name):
                let op = try build(subExpr)
                op.name = name
                return op

            case let .scalarComplement(lhs, rhs):
                let op = try build(rhs)
                assn = Assignment(name: newName(),
                                  shape: op.shape,
                                  rValue: .scalarComplement(lhs, op),
                                  graph: self)
            }
                

            /// Add assignment to the tape
            tape.append(assn)
            return assn
        }
        
        try build(expression)
    }
    
}
