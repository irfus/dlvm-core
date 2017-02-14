//
//  BasicBlock.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

import Foundation

open class BasicBlock : IRCollection, Named {

    public typealias Element = Instruction

    /// Name of the basic block
    open var name: String

    /// Parent function
    open internal(set) weak var parent: Function?

    /// Module containing the parent function
    open weak var module: Module? {
        return parent?.parent
    }

    ///
    /// ## Instructions
    ///

    /// Set of ordered instructions
    fileprivate let instructionSet = NSMutableOrderedSet()
    /// Table of defining(named) instructions for name lookup
    fileprivate var operationTable: [String : Def<Operation>] = [:]

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

    open var isForwardEntry: Bool {
        return parent?.forwardEntry === self
    }
    
    open var isBackwardEntry: Bool {
        return parent?.backwardEntry === self
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
            if case let .operation(oper) = inst as! Instruction {
                operationTable[oper.name] = oper
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

    open func containsInstruction(named name: String) -> Bool {
        return operationTable.keys.contains(name)
    }

    /// Append the instruction to the basic block
    open func append(_ instruction: Instruction) {
        instructionSet.add(instruction)
        if case let .operation(operation) = instruction {
            operationTable[operation.name] = operation
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
        if case let .operation(operation) = instruction {
            operationTable.removeValue(forKey: operation.name)
        }
    }

    /// Returns the instruction having the specified name 
    /// in the current basic block
    open func instruction(named name: String) -> Def<Operation>? {
        return operationTable[name]
    }

}

// MARK: - Analysis information
extension BasicBlock {

    /// Update analysis information
    func updateAnalysisInformation() {
        /// Update users
        updateUsers()
    }

    /// Update user information
    private func updateUsers() {
        for inst in instructions {
            if case let .operation(oper) = inst {
                oper.removeAllUsers()
            }
        }
    }

}
