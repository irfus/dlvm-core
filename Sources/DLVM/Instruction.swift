//
//  Instruction.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

public class Variable {
    public enum Kind {
        case scalar(DataType)
        case tensor(TensorShape)
    }
    
    public let name: String
    public let id: Int
    public let dataType: DataType
    public let kind: Kind
    public let instruction: Instruction
    
    init(name: String, id: Int, dataType: DataType,
         kind: Kind, instruction: Instruction) {
        self.name = name
        self.dataType = dataType
        self.kind = kind
        self.id = id
        self.instruction = instruction
    }
}

public enum ActivationFunction {
    case sigmoid, relu, tanh
}

public indirect enum Instruction {
    case input(shape: TensorShape)
    case parameter(shape: TensorShape)
    case negate(Variable)
    case add(Variable, Variable)
    case mul(Variable, Variable)
    case min(Variable, Variable)
    case max(Variable, Variable)
    case dotProduct(Variable, Variable)
    case product(Variable, Variable)
    case activation(ActivationFunction, Variable)
    case concat(Variable)
    case phi([Variable])
    case condBranch(Variable, BasicBlock, BasicBlock)
}
