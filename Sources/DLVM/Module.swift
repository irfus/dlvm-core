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
    
    var declarations: [String : Variable] = [:]
    let basicBlocks = NSMutableOrderedSet()

    public var elements: [BasicBlock] {
        return basicBlocks.array as! [BasicBlock]
    }

    public init() {}

    public init(declarations: [Variable], basicBlocks: [BasicBlock]) {
        for decl in declarations {
            self.declarations[decl.name] = decl
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

// MARK: - Variables
public extension Module {

    public func declare(_ variable: Variable) {
        declarations[variable.name] = variable
    }

    public func declaration(named name: String) -> Variable? {
        return declarations[name]
    }

    public func removeDeclaration(_ variable: Variable) {
        declarations.removeValue(forKey: variable.name)
    }

    @discardableResult
    public func removeDeclaration(named name: String) -> Variable? {
        let variable = declarations[name]
        declarations.removeValue(forKey: name)
        return variable
    }

}
