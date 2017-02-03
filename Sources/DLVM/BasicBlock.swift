//
//  BasicBlock.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

import Foundation

open class BasicBlock : IRCollection, IRObject {

    public typealias Element = Instruction

    /// Name of the basic block
    open var name: String

    /// Parent module
    open internal(set) weak var module: Module?

    ///
    /// ## Instructions
    ///

    /// Set of ordered instructions
    fileprivate let instructionSet = NSMutableOrderedSet()
    /// Table of defining(named) instructions for name lookup
    fileprivate var instructionTable: [String : DefiningInstruction] = [:]

    /// Instruction list
    /// - Note: this is an API getter that returns a facade object
    /// from instructionSet
    open var instructions: [Instruction] {
        #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
            return instructionSet.array as! [Instruction]
        #else
            return instructionSet.map { $0 as! Instruction }
        #endif
    }

    ///
    /// ## Basic block hierarchy
    ///

    /// Parent basic block in the hierarchy
    open fileprivate(set) weak var parent: BasicBlock? {
        didSet {
            module = parent?.module
        }
    }

    /// Child basic blocks in the hierarchy
    fileprivate var childSet = NSMutableOrderedSet()
    /// Table of child basic blocks for name lookup
    fileprivate var childTable: [String : BasicBlock] = [:]

    /// Child basic block list
    /// - Note: this is an API getter that returns a facade object
    /// from instructionSet
    open var children: [BasicBlock] {
        #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
            return childSet.array as! [BasicBlock]
        #else
            return childSet.map { $0 as! BasicBlock }
        #endif
    }

    ///
    /// ## Extensions
    ///

    /// Extension type of the basic block
    public enum ExtensionType {
        case backpropagation
    }

    /// Extension type
    open fileprivate(set) var extensionType: ExtensionType?
    /// Basic block extensions
    open fileprivate(set) var extensions: [ExtensionType : BasicBlock] = [:]

    /// Main basic block representing the corresponding forward pass
    open weak fileprivate(set) var mainBlock: BasicBlock? {
        didSet {
            module = mainBlock?.module
        }
    }

    /// Whether this basic block is an extension of another
    open var isExtension: Bool {
        return extensionType != nil
    }
    
    public required init(name: String) {
        self.name = name
    }

    public convenience init(name: String, instructions: [Instruction]) {
        self.init(name: name)
        self.instructionSet.addObjects(from: instructions)
        updateInstructionTable()
    }

    internal func updateInstructionTable() {
        for inst in instructionSet {
            if let defInst = inst as? DefiningInstruction {
                instructionTable[defInst.name] = defInst
            }
        }
    }

}

// MARK: - Child basic blocks
// - Note: Child basic blocks are fully managed by BasicBlock class
extension BasicBlock {

    open func makeChild(named name: String) -> Self {
        let block = type(of: self).init(name: name)
        addChild(block)
        return block
    }

    open func addChild(_ child: BasicBlock) {
        guard !containsChild(child) else { return }
        child.parent = self
        childSet.add(child)
        childTable[child.name] = child
    }

    open func containsChild(_ child: BasicBlock) -> Bool {
        return childSet.contains(child)
    }

    open func removeChild(_ child: BasicBlock) {
        precondition(hasChild(child), "This basic block is not a child")
        childSet.remove(child)
        childTable.removeValue(forKey: child.name)
    }

    @discardableResult
    open func removeChild(named name: String) -> BasicBlock? {
        guard let child = child(named: name) else {
            return nil
        }
        removeChild(child)
        return child
    }

    open func hasChild(_ child: BasicBlock) -> Bool {
        return childSet.contains(child)
    }

    open func child(named name: String) -> BasicBlock? {
        return childTable[name]
    }

    open func containsChild(named name: String) -> Bool {
        return childTable.keys.contains(name)
    }

    /// All descendants, exhaustively, post-order
    /// - Complexity: O(n^2)
    open var descendants: [BasicBlock] {
        return children.flatMap { $0.descendants + [$0] }
    }

    /// Exhaustively search for and return descendant
    /// - Complexity: O(n^2)
    open func descendant(named name: String) -> BasicBlock? {
        return child(named: name)
            ?? children.lazy.flatMap{$0.descendant(named: name)}.first
    }

    /// Exhaustively search for and return descendant
    /// - Complexity: O(n^2)
    open func hasDescendant(named name: String) -> Bool {
        return containsChild(named: name)
            || children.lazy.contains(where: {$0.hasDescendant(named: name)})
    }

}

// MARK: - IRCollection
extension BasicBlock {

    /// Same as instructions
    open var elements: [Instruction] {
        return instructions
    }

    open func contains(_ element: Instruction) -> Bool {
        return instructionSet.contains(element)
    }

    open func containsInstruction(named name: String) -> Bool {
        return instructionTable.keys.contains(name)
    }

    /// Append the instruction to the basic block
    open func append(_ instruction: Instruction) {
        instructionSet.add(instruction)
        instruction.parent = self
        if let instruction = instruction as? DefiningInstruction {
            instructionTable[instruction.name] = instruction
        }
        if let instruction = instruction as? NestingInstruction {
            addChild(instruction.body)
        }
    }

    /// Index of the instruction in the basic block
    open func index(of instruction: Instruction) -> Int? {
        return instructionSet.index(of: instruction)
    }

    /// Remove the instruction from the basic block
    ///
    /// - Precondition: instruction is in the basic block
    open func remove(_ instruction: Instruction) {
        precondition(contains(instruction),
                     "Instruction is not in the basic block")
        instructionSet.remove(instruction)
        instruction.parent = nil
        if let instruction = instruction as? DefiningInstruction {
            instructionTable.removeValue(forKey: instruction.name)
        }
        if let instruction = instruction as? NestingInstruction {
            removeChild(instruction.body)
        }
    }

    /// Returns the instruction having the specified name 
    /// in the current basic block
    open func instruction(named name: String) -> DefiningInstruction? {
        return instructionTable[name]
    }

}

// MARK: - Hierarchical basic block properties
extension BasicBlock {
    
    open var depth: Int {
        return parent?.depth.advanced(by: 1) ?? 0
    }

    open var root: BasicBlock {
        return parent?.root ?? self
    }

    /// Returns the instruction having the specified name
    /// in the current scope (including self, forwardBlock?, and parent?)
    /// - Complexity: O(depth)
    open func contextualInstruction(named name: String) -> DefiningInstruction? {
        return instruction(named: name)
            /// Search forward basic block
            ?? mainBlock?.contextualInstruction(named: name)
            /// Search parent
            ?? parent?.contextualInstruction(named: name)
    }

    open func contextualValue(named name: String) -> Value? {
        return module?.globalValue(named: name) ?? contextualInstruction(named: name)
    }

    /// Search the module for and returns the global value having the specified name
    open func globalValue(named name: String) -> Value? {
        return module?.globalValue(named: name)
    }

}

// MARK: - Basic block extensions
extension BasicBlock {

    open func makeExtension(ofType type: ExtensionType) -> BasicBlock {
        let bb = BasicBlock(name: name)
        self[type] = bb
        return bb
    }

    open func hasExtension(ofType type: ExtensionType) -> Bool {
        return extensions.keys.contains(type)
    }

    /// Set/get an extension
    open subscript(extensionType: ExtensionType) -> BasicBlock? {
        get {
            return extensions[extensionType]
        }
        set {
            guard let newValue = newValue else { return }
            extensions[extensionType] = newValue
            newValue.mainBlock = self
            newValue.name = name
            newValue.extensionType = extensionType
        }
    }

}
