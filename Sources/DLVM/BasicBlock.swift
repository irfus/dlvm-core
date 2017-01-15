//
//  BasicBlock.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

import Foundation

open class BasicBlock : IRCollection, IRObject, Differentiable {

    public typealias Element = Instruction

    open var name: String
    open weak var parent: Module?

    open var gradient: BasicBlock?

    /// Take advantange of great data structures from Foundation
    let instructionSet = NSMutableOrderedSet()
    
    /// Same as instructions
    open var elements: [Instruction] {
        return instructions
    }

    open var instructions: [Instruction] {
        return instructionSet.array as! [Instruction]
    }

    public init(name: String) {
        self.name = name
    }

    public convenience init(name: String, instructions: [Instruction]) {
        self.init(name: name)
        self.instructionSet.addObjects(from: instructions)
    }

}

// MARK: - IRCollection
extension BasicBlock {

    public func contains(_ element: Instruction) -> Bool {
        return instructionSet.contains(element)
    }

    /// Append the instruction to the basic block
    open func append(_ instruction: Instruction) {
        instructionSet.add(instruction)
        instruction.parent = self
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
