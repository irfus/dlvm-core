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
    open weak var parent: Module?

    let instructionSet = NSMutableOrderedSet()
    var gradientSet = NSMutableOrderedSet()

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

    open var gradients: [Instruction] {
        #if os(macOS) || os(iOS) || os(watchOS) || os(tvOS)
            return gradientSet.array as! [Instruction]
        #else
            return gradientSet.map { $0 as! Instruction }
        #endif
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

    open func contains(_ element: Instruction) -> Bool {
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

// MARK: - Gradient section
extension BasicBlock {

    open var hasGradient: Bool {
        return gradientSet.count != 0
    }

    open func containsGradient(_ instruction: Instruction) -> Bool {
        return gradientSet.contains(instruction)
    }

    open func appendGradient(_ instruction: Instruction) {
        gradientSet.add(instruction)
    }

    open func index(ofGradient instruction: Instruction) -> Int? {
        return gradientSet.index(of: instruction)
    }

    open func removeGradient(_ instruction: Instruction) {
        precondition(gradientSet.contains(instruction),
                     "Instruction is not in the gradient section of the basic block")
    }

}
