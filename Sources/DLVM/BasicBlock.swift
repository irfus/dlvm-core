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

    open var name: String
    open weak var parent: BasicBlock?

    let instructionSet = NSMutableOrderedSet()

    /// Same as instructions
    open var elements: [Instruction] {
        return instructions
    }

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

    public convenience init(name: String, depth: Int, instructions: [Instruction]) {
        self.init(name: name)
        self.instructionSet.addObjects(from: instructions)
    }

}

// MARK: - Hierarchical basic block properties
extension BasicBlock {

    open var depth: Int {
        return parent?.depth.advanced(by: 1) ?? 0
    }
    
}

// MARK: - IRCollection
extension BasicBlock {

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

}
