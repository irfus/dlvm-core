//
//  Variable.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

public protocol Operand : TextOutputStreamable {}

public protocol ScalarOperand : Operand {
    var type: ScalarType { get }
}

public protocol VariableOperand: class, Operand {
    var name: String { get }
    var definition: VariableProducer? { get }
}

public protocol VariableProducer : class, TextOutputStreamable {
    func makeVariable(named name: String) -> VariableOperand?
}

public enum ImmediateOperand: Operand, ScalarOperand {
    case bool(Bool)
    case int(Int)
    case float(Double)

    public var type: ScalarType {
        switch self {
        case .bool(_): return .bool
        case .int(_): return .int
        case .float(_): return .float
        }
    }
}

open class ScalarVariable : VariableOperand, ScalarOperand {
    public let name: String
    public let type: ScalarType
    public internal(set) var definition: VariableProducer?

    public init(name: String, type: ScalarType,
                definition: VariableProducer?) {
        self.name = name
        self.type = type
        self.definition = definition
        if let inst = definition as? Instruction {
            inst.variable = self
        }
    }
}

open class TensorVariable : VariableOperand {
    public let name: String
    public let dataType: DataType
    public let shape: TensorShape
    public internal(set) var definition: VariableProducer?

    public init(name: String, dataType: DataType,
                shape: TensorShape, definition: VariableProducer?) {
        self.name = name
        self.dataType = dataType
        self.shape = shape
        self.definition = definition
        if let inst = definition as? Instruction {
            inst.variable = self
        }
    }
}
