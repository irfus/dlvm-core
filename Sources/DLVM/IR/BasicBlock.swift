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

    /// Instruction list
    open var elements: OrderedMapSet<Instruction> = []

    /// Parent function
    open unowned var parent: Section

    public internal(set) var predecessors: ObjectSet<BasicBlock> = []

    public required init(name: String, parent: Section) {
        self.name = name
        self.parent = parent
    }

}

// MARK: - Predicates and accessors
public extension BasicBlock {
    
    func containsOperation(_ operation: Def<Operation>) -> Bool {
        return elements.contains(where: { $0.definition === operation })
    }

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

    var isYielding: Bool {
        return terminator?.isYield ?? false
    }

    var isForward: Bool {
        return parent.isForward
    }

    var function: Function {
        return parent.parent
    }

    var module: Module {
        return function.parent
    }

    var isEntry: Bool {
        return parent.entry === self
    }

}
