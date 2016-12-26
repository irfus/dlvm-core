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
    weak var definition: Instruction? { get }
}

public protocol VariableProducer {
    func makeVariable(named name: String) -> Variable
}

open class UnavailableVariable : Variable {
    public static let shared = UnavailableVariable()
    public let name: String = "ε"
    public let definition: Instruction? = nil
    private init() { }
}

public enum Immediate : Operand {
    case bool(Bool)
    case int(Int)
    case float(Double)
}

open class ScalarVariable : Variable {
    public let name: String
    public let type: ScalarType
    public internal(set) weak var definition: Instruction?
    
    public init(name: String, type: ScalarType,
                definition: Instruction? = nil) {
        self.name = name
        self.type = type
        self.definition = definition
    }
}

open class TensorVariable : Variable {
    public let name: String
    public let dataType: DataType
    public let shape: TensorShape
    public internal(set) weak var definition: Instruction?
    
    public init(name: String, dataType: DataType,
                shape: TensorShape, definition: Instruction? = nil) {
        self.name = name
        self.dataType = dataType
        self.shape = shape
        self.definition = definition
    }
}

public enum ActivationFunction {
    case sigmoid, relu, tanh
}

public enum TransformationFunction {
    case log, softmax
}
