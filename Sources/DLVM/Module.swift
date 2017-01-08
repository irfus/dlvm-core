//
//  Module.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

import Foundation

/// Module representing a neural network
open class Module {

    public typealias Element = BasicBlock

    open var name: String
    
    var variableTable: [String : VariableOperand] = [:]
    let basicBlockSet = NSMutableOrderedSet()
    var basicBlockTable: [String : BasicBlock] = [:]

    open lazy var entryBlock: BasicBlock? = self.basicBlocks.first

    open var basicBlocks: [BasicBlock] {
        return basicBlockSet.array as! [BasicBlock]
    }

    public init(name: String) {
        self.name = name
    }

    public init(name: String, variables: [VariableOperand],
                basicBlocks: [BasicBlock]) {
        self.name = name
        for variable in variables {
            variableTable[variable.name] = variable
        }
        self.basicBlockSet.addObjects(from: basicBlocks)
        for bb in basicBlocks {
            basicBlockTable[bb.name] = bb
        }
    }
}

extension Module {

    open func append(_ basicBlock: BasicBlock) {
        precondition(!basicBlockTable.keys.contains(basicBlock.name),
                     "Basic block named \(basicBlock.name) already exists")
        basicBlockSet.add(basicBlock)
        basicBlockTable[basicBlock.name] = basicBlock
        basicBlock.parent = self
    }

    open func index(of basicBlock: BasicBlock) -> Int? {
        return basicBlocks.index(of: basicBlock)
    }

    open func remove(_ basicBlock: BasicBlock) {
        basicBlockSet.remove(basicBlock)
        basicBlockTable.removeValue(forKey: basicBlock.name)
        basicBlock.parent = nil
    }

    open func basicBlock(named name: String) -> BasicBlock? {
        return basicBlockTable[name]
    }
    
}

// MARK: - Global variables
extension Module {

    /// Global variables
    open var variables: AnyCollection<VariableOperand> {
        return AnyCollection(variableTable.values)
    }

    open var externalVariables: AnyCollection<VariableOperand> {
        let variables = variableTable.values.filter { $0.definition == nil }
        return AnyCollection(variables)
    }

    open var definedVariables: AnyCollection<VariableOperand> {
        let variables = variableTable.values.filter { $0.definition != nil }
        return AnyCollection(variables)
    }

    open func add(_ variable: VariableOperand) {
        precondition(!variableTable.keys.contains(variable.name),
                     "Variable named \(variable.name) already exists")
        precondition(!(variable.definition is Instruction),
                     "Global variable definition cannot be an instruction")
        variableTable[variable.name] = variable
    }

    open func variable(named name: String) -> VariableOperand? {
        return variableTable[name]
    }

    open func remove(_ variable: VariableOperand) {
        variableTable.removeValue(forKey: variable.name)
    }

    @discardableResult
    open func removeVariable(named name: String) -> VariableOperand? {
        let variable = variableTable[name]
        variableTable.removeValue(forKey: name)
        return variable
    }

}
