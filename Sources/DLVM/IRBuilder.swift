//
//  IRBuilder.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

public class IRBuilder {

    public let module: Module
    public internal(set) var currentBlock: BasicBlock?

    fileprivate var globalNameId: Int = 0
    fileprivate var nameIdTable: [String : Int] = [:]

    public init() {
        module = Module()
    }

}

public extension IRBuilder {

    public func makeName() -> String {
        return disambiguatedName(for: "v\(globalNameId)")
    }

    public func disambiguatedName(for name: String) -> String {
        if let id = nameIdTable[name] {
            nameIdTable[name] = id + 1
            return name + ".\(id)"
        }
        nameIdTable[name] = 1
        return name
    }
    
}

public extension IRBuilder {

    @discardableResult
    public func beginBasicBlock(named name: String) -> BasicBlock {
        let block = BasicBlock(name: name)
        currentBlock = block
        return block
    }

    @discardableResult
    public func declareTensor(named name: String, dataType: DataType,
                              shape: TensorShape) -> Tensor {
        let tensor = Tensor(name: name, dataType: dataType, shape: shape)
        module.declare(tensor)
        return tensor
    }

    @discardableResult
    public func declareScalar(named name: String,
                              type: ScalarType) -> Scalar {
        let scalar = Scalar(name: name, type: type)
        module.declare(scalar)
        return scalar
    }

    @discardableResult
    public func build(_ instructionKind: Instruction.Kind,
                      named name: String? = nil) -> Variable {
        let instruction = Instruction(kind: instructionKind)
        return instruction.makeVariable(named: name ?? makeName())
    }

}
