//
//  Module.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

import Foundation

/// Module representing a neural network
open class Module : IRCollection {

    public typealias Element = BasicBlock

    open var name: String?
    
    var declarationTable: [String : VariableOperand] = [:]
    let basicBlocks = NSMutableOrderedSet()
    var basicBlockTable: [String : BasicBlock] = [:]

    open lazy var entryBlock: BasicBlock? = self.first

    open var elements: [BasicBlock] {
        return basicBlocks.array as! [BasicBlock]
    }

    public init(name: String? = nil) {
        self.name = name
    }

    public init(name: String? = nil,
                declarations: [VariableOperand], basicBlocks: [BasicBlock]) {
        self.name = name
        for decl in declarations {
            declarationTable[decl.name] = decl
        }
        self.basicBlocks.addObjects(from: basicBlocks)
        for bb in basicBlocks {
            basicBlockTable[bb.name] = bb
        }
    }
}

// MARK: - IRCollection
extension Module {

    open func append(_ basicBlock: BasicBlock) {
        basicBlocks.add(basicBlock)
        basicBlockTable[basicBlock.name] = basicBlock
        basicBlock.parent = self
    }

    open func index(of basicBlock: BasicBlock) -> Int? {
        return elements.index(of: basicBlock)
    }

    open func remove(_ basicBlock: BasicBlock) {
        basicBlocks.remove(basicBlock)
        basicBlockTable.removeValue(forKey: basicBlock.name)
        basicBlock.parent = nil
    }

    open func basicBlock(named name: String) -> BasicBlock? {
        return basicBlockTable[name]
    }
    
}

// MARK: - Declarations (global variables)
extension Module {

    open var declarations: AnyCollection<VariableOperand> {
        return AnyCollection(declarationTable.values)
    }

    open func addDeclaration(_ variable: VariableOperand) {
        declarationTable[variable.name] = variable
    }

    open func declaration(named name: String) -> VariableOperand? {
        return declarationTable[name]
    }

    open func removeDeclaration(_ variable: VariableOperand) {
        declarationTable.removeValue(forKey: variable.name)
    }

    @discardableResult
    open func removeDeclaration(named name: String) -> VariableOperand? {
        let variable = declarationTable[name]
        declarationTable.removeValue(forKey: name)
        return variable
    }

}
