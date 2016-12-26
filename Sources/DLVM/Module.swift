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
    
    var declarationTable: [String : Variable] = [:]
    let basicBlocks = NSMutableOrderedSet()

    public var elements: [BasicBlock] {
        return basicBlocks.array as! [BasicBlock]
    }

    public init() {}

    public init(declarations: [Variable], basicBlocks: [BasicBlock]) {
        for decl in declarations {
            self.declarationTable[decl.name] = decl
        }
        self.basicBlocks.addObjects(from: basicBlocks)
    }
}

// MARK: - IRCollection
extension Module {

    public func append(_ basicBlock: BasicBlock) {
        basicBlocks.add(basicBlock)
        basicBlock.parent = self
    }

    public func index(of basicBlock: BasicBlock) -> Int? {
        return elements.index(of: basicBlock)
    }

    public func remove(_ basicBlock: BasicBlock) {
        basicBlocks.remove(basicBlock)
    }
    
}

// MARK: - Declarations (global variables)
public extension Module {

    public var declarations: AnyCollection<Variable> {
        return AnyCollection(declarationTable.values)
    }

    public func addDeclaration(_ variable: Variable) {
        declarationTable[variable.name] = variable
    }

    public func declaration(named name: String) -> Variable? {
        return declarationTable[name]
    }

    public func removeDeclaration(_ variable: Variable) {
        declarationTable.removeValue(forKey: variable.name)
    }

    @discardableResult
    public func removeDeclaration(named name: String) -> Variable? {
        let variable = declarationTable[name]
        declarationTable.removeValue(forKey: name)
        return variable
    }

}
