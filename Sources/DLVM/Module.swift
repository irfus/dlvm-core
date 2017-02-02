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

    /// Global values
    fileprivate var inputTable: [String : Input] = [:]
    fileprivate var parameterTable: [String : Parameter] = [:]
    fileprivate var outputTable: [String : Output] = [:]

    /// Global basic blocks
    fileprivate let basicBlockSet = NSMutableOrderedSet()
    fileprivate var basicBlockTable: [String : BasicBlock] = [:]

    open lazy var entryBlock: BasicBlock? = self.basicBlocks.first
    
    open var basicBlocks: [BasicBlock] {
        #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
            return basicBlockSet.array as! [BasicBlock]
        #else
            return basicBlockSet.map { $0 as! BasicBlock }
        #endif
    }
    
    public init(name: String) {
        self.name = name
    }
    
    public init(name: String, inputs: [Input], parameters: [Parameter],
                basicBlocks: [BasicBlock]) {
        self.name = name
        /// Add inputs
        for input in inputs {
            inputTable[name] = input
        }
        /// Add parameters
        for parameter in parameters {
            parameterTable[name] = parameter
        }
        /// Add basic blocks
        self.basicBlockSet.addObjects(from: basicBlocks)
        for bb in basicBlocks {
            basicBlockTable[bb.name] = bb
        }
    }
}

// MARK: - Basic block
extension Module {
    
    open func insert(_ basicBlock: BasicBlock) {
        if let existingBlock = self.basicBlock(named: basicBlock.name) {
            remove(existingBlock)
        }
        basicBlockSet.add(basicBlock)
        basicBlockTable[basicBlock.name] = basicBlock
        basicBlock.module = self
    }

    open func index(of basicBlock: BasicBlock) -> Int? {
        return basicBlocks.index(of: basicBlock)
    }
    
    open func remove(_ basicBlock: BasicBlock) {
        basicBlockSet.remove(basicBlock)
        basicBlockTable.removeValue(forKey: basicBlock.name)
        basicBlock.module = self
    }
    
    open func basicBlock(named name: String) -> BasicBlock? {
        return basicBlockTable[name]
    }

    open func containsBasicBlock(named name: String) -> Bool {
        return basicBlockTable.keys.contains(name)
    }

    open func contains(_ basicBlock: BasicBlock) -> Bool {
        return basicBlockSet.contains(basicBlock)
    }

}

// MARK: - Global values
extension Module {

    open func insert(_ value: GlobalValue) {
        switch value {
        case let input as Input:
            inputTable[input.name] = input
            input.parent = self
        case let parameter as Parameter:
            parameterTable[parameter.name] = parameter
            parameter.parent = self
        case let output as Output:
            outputTable[output.name] = output
            output.parent = self
        default:
            preconditionFailure("Unsupported global value")
        }
    }

    open func remove(_ value: GlobalValue) {
        switch value {
        case let input as Input:
            inputTable[value.name] = nil
            input.parent = nil
        case let parameter as Parameter:
            parameterTable[parameter.name] = parameter
            parameter.parent = nil
        case let output as Output:
            outputTable[output.name] = output
            output.parent = nil
        default:
            preconditionFailure("Unsupported global value")
        }
    }

    open func globalValue(named name: String) -> Value? {
        return inputTable[name] ?? parameterTable[name] ?? outputTable[name]
    }

    open func containsGlobalValue(named name: String) -> Bool {
        return inputTable.keys.contains(name)
            || parameterTable.keys.contains(name)
            || outputTable.keys.contains(name)
    }

}

// MARK: - Input
extension Module {
    
    /// Global inputs
    open var inputs: AnyCollection<Input> {
        return AnyCollection(inputTable.values)
    }
    
    open func input(named name: String) -> Input? {
        return inputTable[name]
    }
    
    @discardableResult
    open func removeInput(named name: String) -> Input? {
        let input = inputTable[name]
        inputTable.removeValue(forKey: name)
        return input
    }

    open func containsInput(named name: String) -> Bool {
        return inputTable.keys.contains(name)
    }
    
}

// MARK: - Output
extension Module {
    
    /// Global outputs
    open var outputs: AnyCollection<Output> {
        return AnyCollection(outputTable.values)
    }

    open func output(named name: String) -> Output? {
        return outputTable[name]
    }
    
    @discardableResult
    open func removeOutput(named name: String) -> Output? {
        let output = outputTable[name]
        outputTable.removeValue(forKey: name)
        return output
    }

    open func containsOutput(named name: String) -> Bool {
        return outputTable.keys.contains(name)
    }
    
}

// MARK: - Parameter
extension Module {
    
    /// Global inputs
    open var parameters: AnyCollection<Parameter> {
        return AnyCollection(parameterTable.values)
    }
    
    open func parameter(named name: String) -> Parameter? {
        return parameterTable[name]
    }
    
    @discardableResult
    open func removeParameter(named name: String) -> Parameter? {
        let parameter = parameterTable[name]
        parameterTable.removeValue(forKey: name)
        return parameter
    }

    open func containsParameter(named name: String) -> Bool {
        return parameterTable.keys.contains(name)
    }
    
}
