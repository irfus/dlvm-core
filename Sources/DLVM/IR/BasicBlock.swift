//
//  BasicBlock.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

import Foundation

open class BasicBlock : IRCollection, Named, Global {

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

    /// Whether this basic block is an extension of another
    open var isExtension: Bool {
        return extensionType != nil
    }

    /// Whether this basic is an entry block
    /// - Note: This depends on the parent module's entry block; if this block
    /// is not added to a module or doesn't have the name "module", then it's
    /// not considered an entry
    open var isEntry: Bool {
        return module?.entryBlock === self
    }

    public required init(name: String) {
        self.name = name
    }

    public convenience init(name: String, instructions: [Instruction]) {
        self.init(name: name)
        self.instructionSet.addObjects(from: instructions)
        updateInstructionTable()
    }

    private func updateInstructionTable() {
        for inst in instructionSet {
            if let defInst = inst as? DefiningInstruction {
                instructionTable[defInst.name] = defInst
            }
        }
    }

    ///
    /// ## Analysis information
    ///

    /// Operands used in this immediate basic block
    open fileprivate(set) var exportedOutputs: NamedObjectSet<Output> = []

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
        if let instruction = instruction as? ExportInstruction {
            exportedOutputs.insert(instruction.destination)
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
    }

    /// Returns the instruction having the specified name 
    /// in the current basic block
    open func instruction(named name: String) -> DefiningInstruction? {
        return instructionTable[name]
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

    open func removeExtension(ofType type: ExtensionType) {
        extensions.removeValue(forKey: type)
    }

    /// Set/get an extension
    open subscript(extensionType: ExtensionType) -> BasicBlock? {
        get {
            return extensions[extensionType]
        }
        set {
            guard let newValue = newValue else { return }
            extensions[extensionType] = newValue
            newValue.name = name
            newValue.extensionType = extensionType
        }
    }

}

// MARK: - Analysis information
extension BasicBlock {

    /// Update analysis information
    func updateAnalysisInformation() {
        /// Clear used global values
        exportedOutputs.removeAll()
        /// Update users
        updateUsers()
    }

    /// Update user information
    private func updateUsers() {
        for inst in instructions {
            if let user = inst as? ManagedUsee {
                user.removeAllUsers()
            }
            inst.updateUsers()
            /// Update exported output set
            if let exp = inst as? ExportInstruction {
                exportedOutputs.insert(exp.destination)
            }
        }
    }
    
}
