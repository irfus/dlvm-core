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
    
    var inputTable: [String : Input] = [:]
    var parameterTable: [String : Parameter] = [:]
    let basicBlockSet = NSMutableOrderedSet()
    var basicBlockTable: [String : BasicBlock] = [:]

    open lazy var entryBlock: BasicBlock? = self.basicBlocks.first

    open var basicBlocks: [BasicBlock] {
        return basicBlockSet.array as! [BasicBlock]
    }

    public init(name: String) {
        self.name = name
    }

    public init(name: String, inputs: [Input], parameters: [Parameter],
                basicBlocks: [BasicBlock]) {
        self.name = name
        /// Add inputs
        for input in inputs {
            if let name = input.name {
                inputTable[name] = input
            }
        }
        /// Add parameters
        for parameter in parameters {
            if let name = parameter.name {
                parameterTable[name] = parameter
            }
        }
        /// Add basic blocks
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

    /// Global inputs
    open var inputs: AnyCollection<Input> {
        return AnyCollection(inputTable.values)
    }

    open func add(_ input: Input) {
        if let name = input.name { /// Ignore nameless inputs
            inputTable[name] = input
        }
    }

    open func input(named name: String) -> Input? {
        return inputTable[name]
    }

    open func remove(_ input: Input) {
        if let name = input.name {
            inputTable.removeValue(forKey: name)
        }
    }

    @discardableResult
    open func removeVariable(named name: String) -> Input? {
        let input = inputTable[name]
        inputTable.removeValue(forKey: name)
        return input
    }

}

// MARK: - Global variables
extension Module {

    /// Global inputs
    open var parameters: AnyCollection<Parameter> {
        return AnyCollection(parameterTable.values)
    }

    open func add(_ parameter: Parameter) {
        if let name = parameter.name { /// Ignore nameless parameters
            parameterTable[name] = parameter
        }
    }

    open func parameter(named name: String) -> Parameter? {
        return parameterTable[name]
    }

    open func remove(_ parameter: Parameter) {
        if let name = parameter.name {
            parameterTable.removeValue(forKey: name)
        }
    }

    @discardableResult
    open func removeVariable(named name: String) -> Parameter? {
        let parameter = parameterTable[name]
        parameterTable.removeValue(forKey: name)
        return parameter
    }

}
