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

    public fileprivate(set) var basicBlocks = OrderedNamedObjectSet<BasicBlock>()
    public fileprivate(set) var globals = OrderedNamedObjectSet<Global>()

    open var entryBlock: BasicBlock? {
        return basicBlock(named: "entry")
    }

    public init(name: String) {
        self.name = name
    }
}

// MARK: - Basic block
extension Module {
    
    open func insert(_ basicBlock: BasicBlock) {
        if let existingBlock = self.basicBlock(named: basicBlock.name) {
            remove(existingBlock)
        }
        basicBlocks.insert(basicBlock)
        basicBlock.module = self
    }

    open func index(of basicBlock: BasicBlock) -> Int? {
        return basicBlocks.index(of: basicBlock)
    }
    
    open func remove(_ basicBlock: BasicBlock) {
        basicBlocks.remove(basicBlock)
        basicBlock.module = self
    }

    open func basicBlock(named name: String) -> BasicBlock? {
        return basicBlocks.element(named: name)
    }

    open func containsBasicBlock(named name: String) -> Bool {
        return basicBlocks.containsValue(named: name)
    }

    open func contains(_ basicBlock: BasicBlock) -> Bool {
        return basicBlocks.contains(basicBlock)
    }

}

// MARK: - Globals
extension Module {

    open func insert(_ global: Global) {
        globals.insert(global)
    }

    open func index(of global: Global) -> Int? {
        return globals.index(of: global)
    }
    
    open func remove(_ global: Global) {
        globals.remove(global)
    }
    
    open func global(named name: String) -> Global? {
        return globals.element(named: name)
    }

    open func contains(_ global: Global) -> Bool {
        return globals.contains(global)
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
