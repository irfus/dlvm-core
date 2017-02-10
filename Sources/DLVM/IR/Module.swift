//
//  Module.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

import Foundation

/// Module representing a neural network
open class Module : Named {
    public typealias Element = BasicBlock
    
    open var name: String

    /// Globals
    fileprivate var inputTable: [String : Input] = [:]
    fileprivate var parameterTable: [String : Parameter] = [:]
    fileprivate var outputTable: [String : Output] = [:]
    fileprivate var constantTable: [String : Constant] = [:]

    /// Global basic blocks
    fileprivate let basicBlockSet = NSMutableOrderedSet()
    fileprivate var basicBlockTable: [String : BasicBlock] = [:]

    open var entryBlock: BasicBlock? {
        return basicBlock(named: "entry")
    }
    
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

    open func insert(_ input: Input) {
        inputTable[input.name] = input
        input.module = self
    }

    open func insert(_ parameter: Parameter) {
        parameterTable[parameter.name] = parameter
        parameter.module = self
    }

    open func insert(_ output: Output) {
        outputTable[output.name] = output
        output.module = self
    }

    open func insert(_ constant: Constant) {
        constantTable[constant.name] = constant
        constant.module = self
    }

    open func remove(_ input: Input) {
        inputTable[input.name] = nil
        input.module = nil
    }

    open func remove(_ parameter: Parameter) {
        parameterTable[parameter.name] = nil
        parameter.module = nil
    }

    open func remove(_ output: Output) {
        outputTable[output.name] = nil
        output.module = nil
    }

    open func remove(_ constant: Constant) {
        constantTable[constant.name] = nil
        constant.module = nil
    }

    open func globalValue(named name: String) -> GlobalValue? {
        return parameterTable[name] ?? constantTable[name]
    }

    open func containsGlobalValue(named name: String) -> Bool {
        return parameterTable.keys.contains(name)
            || constantTable.keys.contains(name)
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

// MARK: - Constants
extension Module {

    open var constants: AnyCollection<Constant> {
        return AnyCollection(constantTable.values)
    }

    open func constant(named name: String) -> Constant? {
        return constantTable[name]
    }

    @discardableResult
    open func removeConstant(named name: String) -> Constant? {
        let constant = constantTable[name]
        constantTable.removeValue(forKey: name)
        return constant
    }

    open func containsConstant(named name: String) -> Bool {
        return constantTable.keys.contains(name)
    }
    
}

// MARK: - Analysis information
extension Module {

    open func updateAnalysisInformation() {
        inputs.forEach { $0.removeAllUsers() }
        parameters.forEach { $0.removeAllUsers() }
        outputs.forEach { $0.removeAllUsers() }
        for bb in basicBlocks {
            bb.updateAnalysisInformation()
        }
    }
    
}

// MARK: - Output
extension Module {

    open func write(toFile path: String) throws {
        var contents = ""
        write(to: &contents)
        try contents.write(toFile: path, atomically: true, encoding: .utf8)
    }
    
}
