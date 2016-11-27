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
import CuBLAS

/// Computation graph of the neural network
/// - note: Create a generic graph without CPU/GPU dependencies
/// - parameter DataType: type of elements of the tensor (Float, Double, ...)
public class Graph<DataType : TensorDataProtocol> {
    /// Computation tape (SSA form)
    var tape: [Variable<DataType>] = []

    /// Root of the computation graph
    let root: Expression<DataType>

    /// cuDNN instance
    let dnn: DNN

    /// cuBLAS instance
    let blas: BLAS

    /// Device on which the graph is executed
    let device: Device

    /// Op tensor
    let tensorOperators: TensorOperators<DataType>

    /// Initialize from an expression
    /// - parameter expression: neural network expression
    public init(expression: Expression<DataType>, device: Device = Device.current) throws {
        root = expression
        dnn = DNN.shared(on: device)
        blas = BLAS.global(on: device)
        tensorOperators = TensorOperators()
        self.device = device
        try buildIR(from: expression)
    }

    /// Preallocate data for the assignment form
    func preallocateData() {
        tape.forEach { _ = $0.data; _ = $0.gradient }
    }
}

/// RValue of assignment form
indirect enum RValue<DataType: TensorDataProtocol> {
    case input(shape: TensorShape)
    case parameter(shape: TensorShape, initial: TensorInitializer<DataType>)
    case log(Variable<DataType>)
    case sigmoid(Variable<DataType>)
    case relu(Variable<DataType>)
    case tanh(Variable<DataType>)
    case softmax(Variable<DataType>)
    case negative(Variable<DataType>)
    case add(Variable<DataType>, Variable<DataType>)
    case mul(Variable<DataType>, Variable<DataType>)
    case min(Variable<DataType>, Variable<DataType>)
    case max(Variable<DataType>, Variable<DataType>)
    case product(Variable<DataType>, Variable<DataType>)
    case scalarComplement(DataType, Variable<DataType>)
}

/// Assignment in SSA form (phi-function is not currently implemented)
class Variable<DataType: TensorDataProtocol> {
    var name: String
    let shape: TensorShape
    let rValue: RValue<DataType>
    unowned let graph: Graph<DataType>

    lazy var data: Tensor<DataType> = {
        Device.current = self.graph.device
        switch self.rValue {
        case let .input(shape: shape):
            return Tensor(shape: shape, repeating: .zero, device: self.graph.device)
        case let .parameter(shape: shape, initial: initial):
            switch initial {
            case .zeros:
                return Tensor(shape: shape, repeating: .zero, device: self.graph.device)
            case let .random(from: lowerBound, to: upperBound):
                return Tensor(shape: shape, device: self.graph.device, factory: {
                    DataType.random(from: lowerBound, to: upperBound)
                })
            }
        default:
            return Tensor(shape: self.shape, repeating: .zero, device: self.graph.device)
        }
    }()

    lazy var gradient: Tensor<DataType> = {
        return Tensor(shape: self.shape, repeating: .zero, device: self.graph.device)
    }()

    /// Initialize a variable
    ///
    /// - Parameters:
    ///   - name: name of the variable
    ///   - shape: shape of the tensor representing nodes
    ///   - rValue: source of assignment
    ///   - graph: unowned reference to the graph
    init(name: String, shape: TensorShape, rValue: RValue<DataType>, graph: Graph<DataType>) {
        self.name = name
        self.shape = shape
        self.rValue = rValue
        self.graph = graph
    }
}

/// Graph creation error
public enum GraphError : Error {
    case productDimensionMismatch(TensorShape, TensorShape)
}

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
        func build(_ node: Expression<DataType>) throws -> Variable<DataType> {
            let assn: Variable<DataType>
            switch node {
            case let .input(shape: shape, name: name):
                assn = Variable(name: name ?? newName(),
                                shape: shape,
                                rValue: .input(shape: shape),
                                graph: self)
                
            case let .parameter(shape: shape, initial: initializer, name: name):
                assn = Variable(name: name ?? newName(),
                                shape: shape,
                                rValue: .parameter(shape: shape, initial: initializer),
                                graph: self)
                
            case let .log(x):
                let op = try build(x)
                assn = Variable(name: newName(),
                                shape: op.shape,
                                rValue: .log(op),
                                graph: self)
                
            case let .tanh(x):
                let op = try build(x)
                assn = Variable(name: newName(),
                                shape: op.shape,
                                rValue: .tanh(op),
                                graph: self)
                
            case let .relu(x):
                let op = try build(x)
                assn = Variable(name: newName(),
                                shape: op.shape,
                                rValue: .relu(op),
                                graph: self)
                
            case let .softmax(x):
                let op = try build(x)
                assn = Variable(name: newName(),
                                shape: op.shape,
                                rValue: .softmax(op),
                                graph: self)
                
            case let .sigmoid(x):
                let op = try build(x)
                assn = Variable(name: newName(),
                                shape: op.shape,
                                rValue: .sigmoid(op),
                                graph: self)
                
            case let .negative(x):
                let op = try build(x)
                assn = Variable(name: newName(),
                                shape: op.shape,
                                rValue: .sigmoid(op),
                                graph: self)
                
            case let .add(lhs, rhs):
                let leftOp = try build(lhs), rightOp = try build(rhs)
                assn = Variable(name: newName(),
                                shape: leftOp.shape,
                                rValue: .add(leftOp, rightOp),
                                graph: self)
                
                
            case let .mul(lhs, rhs):
                let leftOp = try build(lhs), rightOp = try build(rhs)
                assn = Variable(name: newName(),
                                shape: leftOp.shape,
                                rValue: .mul(leftOp, rightOp),
                                graph: self)
                
            case let .min(lhs, rhs):
                let leftOp = try build(lhs), rightOp = try build(rhs)
                assn = Variable(name: newName(),
                                shape: leftOp.shape,
                                rValue: .min(leftOp, rightOp),
                                graph: self)
                
            case let .max(lhs, rhs):
                let leftOp = try build(lhs), rightOp = try build(rhs)
                assn = Variable(name: newName(),
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
                assn = Variable(name: newName(),
                                shape: newShape,
                                rValue: .product(leftOp, rightOp),
                                graph: self)
                
            case let .layer(subExpr, name: name):
                let op = try build(subExpr)
                op.name = name
                return op
                
            case let .scalarComplement(lhs, rhs):
                let op = try build(rhs)
                assn = Variable(name: newName(),
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

extension Variable : Equatable {
    static func ==(lhs: Variable<DataType>, rhs: Variable<DataType>) -> Bool {
        return lhs === rhs
    }
}
