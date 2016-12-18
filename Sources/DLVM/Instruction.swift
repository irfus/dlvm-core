//
//  Instruction.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

public class Variable {

    public enum DataType {
        case int1, int8, int16, int32, int64
        case float8, float16, float32, float64
    }
    
    public let name: String
    public let id: Int
    public let type: DataType
    public let instruction: Instruction
    
    init(name: String, id: Int, type: DataType, instruction: Instruction) {
        self.name = name
        self.type = type
        self.id = id
        self.instruction = instruction
    }
}

/// RValue of assignment form
public indirect enum Instruction {
    case input(shape: TensorShape)
    case parameter(shape: TensorShape)
    case log(Variable)
    case sigmoid(Variable)
    case relu(Variable)
    case tanh(Variable)
    case softmax(Variable)
    case logSoftmax(Variable)
    case negate(Variable)
    case add(Variable, Variable)
    case mul(Variable, Variable)
    case min(Variable, Variable)
    case max(Variable, Variable)
    case dotProduct(Variable, Variable)
    case product(Variable, Variable)
}
