//
//  BasicBlock.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

import Foundation

open class BasicBlock : IRCollection, IRUnit, Named {

    public typealias Element = Instruction

    /// Name of the basic block
    open var name: String

    /// Parent function
    open weak var parent: Function?

    /// Module containing the parent function
    open weak var module: Module? {
        return parent?.parent
    }

    /// Instruction list
    open var elements: OrderedMapSet<Instruction> = []

    open var isEntry: Bool {
        return parent?.entry === self
    }

    public required init(name: String) {
        self.name = name
    }

    public convenience init(name: String, instructions: [Instruction]) {
        self.init(name: name)
        self.elements.append(contentsOf: instructions)
    }

    /// ## Analysis information
    public internal(set) var predecessors: ObjectSet<BasicBlock> = []

}

// MARK: - IRCollection
extension BasicBlock {

    open func containsOperation(_ operation: Def<Operation>) -> Bool {
        return elements.contains(.operation(operation))
    }

    /// Append the instruction to the basic block
    open func append(_ instruction: Instruction) {
        elements.append(instruction)
        /// If it's a branch instruction, update successor's predecessor set
        for succ in instruction.successors {
            succ.predecessors.insert(self)
        }
    }

    /// Remove the instruction from the basic block
    ///
    /// - Precondition: instruction is in the basic block
    open func remove(_ instruction: Instruction) {
        elements.remove(instruction)
        /// If it's a branch instruction, update successor's predecessor set
        for succ in instruction.successors {
            succ.predecessors.remove(self)
        }
    }

}


// MARK: - Basic block successors
public extension BasicBlock {

    /// Whether there exists a terminator instruction
    /// - Note: a branching instruction in the middle of the basic block
    /// is not considered a terminator
    var hasTerminator: Bool {
        return elements.last?.isTerminator ?? false
    }

    /// Terminator instruction
    var terminator: Instruction? {
        guard let last = elements.last, last.isTerminator else {
            return nil
        }
        return last
    }

    var successorCount: Int {
        return terminator?.successorCount ?? 0
    }

    var hasSuccessors: Bool {
        return successorCount > 0
    }

    var isReturn: Bool {
        return terminator?.isReturn ?? false
    }

    var isForward: Bool {
        return parent?.forwardPass.contains(self) ?? false
    }

    var isBackward: Bool {
        return parent?.backwardPasses.values.contains(where: {$0.contains(self)}) ?? false
    }

}
