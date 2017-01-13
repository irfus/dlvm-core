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
    
    /// Take advantange of great data structures from Foundation
    let instructions = NSMutableOrderedSet()
    
    open weak var parent: Module?

    open var elements: [Instruction] {
        return instructions.array as! [Instruction]
    }

    public init(name: String) {
        self.name = name
    }

    public init(name: String, instructions: [Instruction]) {
        self.name = name
        self.instructions.addObjects(from: instructions)
    }

}

// MARK: - IRCollection
extension BasicBlock {

    public func contains(_ element: Instruction) -> Bool {
        return instructions.contains(element)
    }

    /// Append the instruction to the basic block
    open func append(_ instruction: Instruction) {
        instructions.add(instruction)
        instruction.parent = self
    }

    /// Index of the instruction in the basic block
    open func index(of instruction: Instruction) -> Int? {
        return instructions.index(of: instruction)
    }

    /// Remove the instruction from the basic block
    ///
    /// - Precondition: instruction is in the basic block
    open func remove(_ instruction: Instruction) {
        precondition(instructions.contains(instruction),
                     "Instruction is not in the basic block")
        instructions.remove(instruction)
        instruction.parent = nil
    }

}
