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
    var outputTable: [String : Output] = [:]
    let basicBlockSet = NSMutableOrderedSet()
    var basicBlockTable: [String : BasicBlock] = [:]
    
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
    
    open func append(_ basicBlock: BasicBlock) {
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
    
    open func add(_ value: GlobalValue) {
        switch value {
        case let input as Input:
            inputTable[input.name] = input
        case let parameter as Parameter:
            parameterTable[parameter.name] = parameter
        case let output as Output:
            outputTable[output.name] = output
        default:
            preconditionFailure("Unsupported global value")
        }
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
    
    open func remove(_ input: Input) {
        inputTable.removeValue(forKey: input.name)
    }
    
    @discardableResult
    open func removeInput(named name: String) -> Input? {
        let input = inputTable[name]
        inputTable.removeValue(forKey: name)
        return input
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
    
    open func remove(_ output: Output) {
        outputTable.removeValue(forKey: output.name)
    }
    
    @discardableResult
    open func removeOutput(named name: String) -> Output? {
        let output = outputTable[name]
        outputTable.removeValue(forKey: name)
        return output
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

    open func remove(_ parameter: Parameter) {
        parameterTable.removeValue(forKey: parameter.name)
    }
    
    @discardableResult
    open func removeParameter(named name: String) -> Parameter? {
        let parameter = parameterTable[name]
        parameterTable.removeValue(forKey: name)
        return parameter
    }
    
}
