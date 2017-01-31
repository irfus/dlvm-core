//
//  BasicBlock.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

import Foundation

open class BasicBlock : IRCollection, IRObject {

    /// Extension type of the basic block
    public enum ExtensionType {
        case backpropagation
    }

    public typealias Element = Instruction

    /// Name of the basic block
    open var name: String

    /// Parent basic block in the hierarchy
    open weak var parent: BasicBlock?

    /// Set of ordered instructions
    fileprivate let instructionSet = NSMutableOrderedSet()
    /// Table of defining(named) instructions for name lookup
    fileprivate var instructionTable: [String : DefiningInstruction] = [:]

    /// Extension type
    open fileprivate(set) var extensionType: ExtensionType?

    /// Main basic block representing the corresponding forward pass
    open weak fileprivate(set) var mainBlock: BasicBlock?

    /// Whether this basic block is an extension of another
    open var isExtension: Bool {
        return extensionType != nil
    }

    /// Basic block extensions
    internal var extensions: [ExtensionType : BasicBlock] = [:]

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
    
    public init(name: String) {
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

// MARK: - IRCollection
extension BasicBlock {

    /// Same as instructions
    open var elements: [Instruction] {
        return instructions
    }

    open func contains(_ element: Instruction) -> Bool {
        return instructionSet.contains(element)
    }

    /// Append the instruction to the basic block
    open func append(_ instruction: Instruction) {
        instructionSet.add(instruction)
        instruction.parent = self
        if let inst = instruction as? LoopInstruction {
            inst.body.parent = self
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
        precondition(instructionSet.contains(instruction),
                     "Instruction is not in the basic block")
        instructionSet.remove(instruction)
        instruction.parent = nil
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

    /// Returns the instruction having the specified name
    /// in the current scope (including self, forwardBlock?, and parent?)
    /// - Complexity: O(depth)
    internal func contextualInstruction(named name: String) -> DefiningInstruction? {
        return instructionTable[name]
            /// Search forward basic block
            ?? mainBlock?.contextualInstruction(named: name)
            /// Search parent
            ?? parent?.contextualInstruction(named: name)
    }
    
}

// MARK: - Basic block extensions
extension BasicBlock {

    /// Set/get an extension
    public subscript(extensionType: ExtensionType) -> BasicBlock? {
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
