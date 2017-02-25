//
//  BasicBlock.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

import Foundation

open class BasicBlock : IRCollection, IRObject, Named {

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

    open var isEntry: Bool {
        return parent?.entry === self
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
            if case let .operation(oper) = (inst as! Instruction).kind {
                operationTable[oper.name] = oper
            }
        }
    }

    /// ## Analysis information
    public internal(set) var predecessors: Set<BasicBlock> = []

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

    open func containsOperation(_ operation: Def<Operation>) -> Bool {
        return instructionSet.contains(Instruction.operation(operation))
    }

    open func containsInstruction(named name: String) -> Bool {
        return operationTable.keys.contains(name)
    }

    /// Append the instruction to the basic block
    open func append(_ instruction: Instruction) {
        instructionSet.add(instruction)
        instruction.parent = self
        if case let .operation(operation) = instruction.kind {
            operationTable[operation.name] = operation
        }
        /// If it's a branch instruction, update successor's predecessor set
        for succ in instruction.successors {
            succ.predecessors.insert(self)
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
        if case let .operation(operation) = instruction.kind {
            operationTable.removeValue(forKey: operation.name)
        }
        /// If it's a branch instruction, update sucessor's predecessor set
        for succ in instruction.successors {
            succ.predecessors.remove(self)
        }
    }

    /// Returns the instruction having the specified name 
    /// in the current basic block
    open func operation(named name: String) -> Def<Operation>? {
        return operationTable[name]
    }

    /// Whether there exists a terminator instruction
    open var hasTerminator: Bool {
        return instructions.last?.isTerminator ?? false
    }

    /// Terminator instruction
    open var terminator: Instruction? {
        guard let last = instructions.last, last.isTerminator else {
            return nil
        }
        return last
    }
    
}

// MARK: - Control Flow Graph
extension BasicBlock : GraphNode {
    public var children: [BasicBlock] {
        guard let terminator = self.terminator else { return [] }
        switch terminator.kind {
        case let .control(.br(dest)):
            return [dest]
        case let .control(.condBr(_, thenBB, elseBB)):
            return [thenBB, elseBB]
        case let .operation(def):
            if case let .pull(_, thenBB, elseBB) = def.value {
                return [thenBB, elseBB]
            }
            fallthrough
        default:
            return []
        }
    }

}
