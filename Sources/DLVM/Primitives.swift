//
//  Instruction.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

public protocol Operand {}

public protocol Variable : Operand {
    var name: String { get }
    weak var definition: Instruction? { get }
    static func ~=(lhs: Variable, rhs: Variable) -> Bool
}

public extension Variable {
    /// Determine if two variables are similar, i.e. have the same
    /// type characteristics
    public static func ~= (lhs: Variable, rhs: Variable) -> Bool {
        switch (lhs, rhs) {
        case let (lhs as ScalarVariable, rhs as ScalarVariable):
            return lhs.type == rhs.type
        case let (lhs as TensorVariable, rhs as TensorVariable):
            return lhs.shape == rhs.shape && lhs.dataType == rhs.dataType
        case (_ as UnavailableVariable, _ as UnavailableVariable):
            return true
        default:
            return false
        }
    }
}

public protocol VariableProducer {
    func makeVariable(named name: String) -> Variable
}

public struct UnavailableVariable : Variable {
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

public struct ScalarVariable : Variable {
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

public struct TensorVariable : Variable {
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
