//
//  Variable.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

public protocol Operand {}

public protocol ScalarOperand : Operand {
    var type: ScalarType { get }
}

public protocol VariableOperand: Operand {
    var name: String { get }
    weak var definition: VariableDefiner? { get }
}

public protocol VariableDefiner : class {}

public protocol VariableProducer {
    func makeVariable(named name: String) -> VariableOperand
}

open class UnavailableVariable : VariableOperand {
    public static let shared = UnavailableVariable()
    public let name: String = "ε"
    public let definition: VariableDefiner? = nil
    private init() { }
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
    public internal(set) weak var definition: VariableDefiner?
    
    public init(name: String, type: ScalarType,
                definition: VariableDefiner? = nil) {
        self.name = name
        self.type = type
        self.definition = definition
    }
}

open class TensorVariable : VariableOperand {
    public let name: String
    public let dataType: DataType
    public let shape: TensorShape
    public internal(set) weak var definition: VariableDefiner?
    
    public init(name: String, dataType: DataType,
                shape: TensorShape, definition: VariableDefiner? = nil) {
        self.name = name
        self.dataType = dataType
        self.shape = shape
        self.definition = definition
    }
}
