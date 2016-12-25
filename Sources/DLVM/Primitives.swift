//
//  Instruction.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

public protocol Operand {}

public protocol Variable : class, Operand {
    var name: String { get }
}

public enum Immediate : Operand {
    case bool(Bool)
    case int(Int)
    case float(Double)
}

public class Scalar : Variable {
    public let name: String
    public let type: ScalarType
    public init(name: String, type: ScalarType) {
        self.name = name
        self.type = type
    }
}

public class Tensor : Variable {
    public let name: String
    public let dataType: DataType
    public let shape: TensorShape
    public init(name: String, dataType: DataType,
                shape: TensorShape) {
        self.name = name
        self.dataType = dataType
        self.shape = shape
    }
}

public enum ActivationFunction {
    case sigmoid, relu, tanh, log
}

public enum TransferFunction {
    case softmax
}

