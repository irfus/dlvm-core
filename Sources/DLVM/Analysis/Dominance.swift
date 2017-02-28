//
//  Dominance.swift
//  DLVM
//
//  Created by Richard Wei on 2/18/17.
//

/// This file contains an implementation of the Cooper-Harvey-Kennedy dominance algorithm
///
/// K. D. Cooper, T. J. Harvey, and K. Kennedy. A simple, fast dominance algorithm.
/// Rice University, 2000.

open class DominatorTree {

    fileprivate var immediateDominators: [Unowned<BasicBlock> : BasicBlock] = [:]
    public unowned let root: BasicBlock

    internal init(root: BasicBlock,
                  immediateDominators: [Unowned<BasicBlock> : BasicBlock] = [:]) {
        self.root = root
        self.immediateDominators = immediateDominators
    }

    /// Initialize from entry node using
    public init(entry: BasicBlock) {
        self.root = entry

        /// Initialization of data flow analysis
        immediateDominators[Unowned(root)] = root

        /// Iteratively build tree
        var changed = true
        repeat {
            changed = false
            for node in root.postorder.reversed().dropFirst() {
                let preds = node.predecessors
                guard var newIDom = preds.first else {
                    preconditionFailure("Successor node doesn't have any predecessor")
                }
                for p in preds.dropFirst() where !contains(p) {
                    newIDom = nearestCommonDominator(p, newIDom)
                }
                if immediateDominator(of: node) !== newIDom {
                    immediateDominators[Unowned(node)] = newIDom
                    changed = true
                }
            }
        } while changed
    }
}

open class PostdominatorTree : DominatorTree {

    /// Initialize from entry node using
    public override init(entry: BasicBlock) {
        super.init(root: entry)

        /// Initialization of data flow analysis
        immediateDominators[Unowned(root)] = root

        /// Iteratively build tree
        var changed = true
        repeat {
            changed = false
            for node in root.transposeTraversed(in: .postorder).reversed().dropFirst() {
                let preds = node.successors
                guard var newIDom = preds.first else {
                    preconditionFailure("Successor node doesn't have any predecessor")
                }
                for p in preds.dropFirst() where !contains(p) {
                    newIDom = nearestCommonDominator(p, newIDom)
                }
                if immediateDominator(of: node) !== newIDom {
                    immediateDominators[Unowned(node)] = newIDom
                    changed = true
                }
            }
        } while changed
    }
    
}

public extension DominatorTree {

    public func immediateDominator(of node: BasicBlock) -> BasicBlock {
        return immediateDominators[Unowned(node)]!
    }

    public func nearestCommonDominator(_ b1: BasicBlock, _ b2: BasicBlock) -> BasicBlock {
        if dominates(b1, b2) { return b1 }
        if dominates(b2, b1) { return b2 }

        /// Collect all b1's dominators
        var b1Dominators: ObjectSet<BasicBlock> = []
        b1Dominators.insert(root)
        var b1Dom = immediateDominator(of: b1)
        while b1Dom !== root {
            b1Dominators.insert(b1Dom)
            b1Dom = immediateDominator(of: b1Dom)
        }

        /// Check all b2's dominators against b1Dominators
        var b2Dom = immediateDominator(of: b2)
        while b2Dom !== root {
            if b1Dominators.contains(b2Dom) {
                return b2Dom
            }
            b2Dom = immediateDominator(of: b2Dom)
        }

        return root
    }

    public func contains(_ node: BasicBlock) -> Bool {
        return immediateDominators.keys.contains(Unowned(node))
    }

    func properlyDominates(_ block: BasicBlock, _ otherBlock: BasicBlock) -> Bool {
        guard block !== otherBlock else { return false }
        if block === root { return true }
        var iDom = immediateDominator(of: otherBlock)
        while iDom !== block && iDom !== root {
            iDom = immediateDominator(of: iDom)
        }
        return iDom !== root
    }

    func dominates(_ block: BasicBlock, _ otherBlock: BasicBlock) -> Bool {
        return block === otherBlock || properlyDominates(block, otherBlock)
    }
}

public extension DominatorTree {
    func immediateDominatorInstruction(of body: BasicBlock) -> Instruction? {
        return immediateDominator(of: body).terminator
    }

    func dominates(_ instruction: Instruction, _ otherInstruction: Instruction) -> Bool {
        let bb1 = instruction.parent, bb2 = otherInstruction.parent
        if bb1 === bb2 {
            return bb1.index(of: instruction)! < bb1.index(of: otherInstruction)!
        }
        return dominates(bb1, bb2)
    }
}

public extension BasicBlock {
    public func isReachable(in domTree: DominatorTree) -> Bool {
        return domTree.contains(self)
    }

    public func dominates(_ other: BasicBlock,
                          in domTree: DominatorTree) -> Bool {
        return domTree.dominates(self, other)
    }
}

public extension Instruction {
    public func dominates(_ other: Instruction,
                          in domTree: DominatorTree) -> Bool {
        return domTree.dominates(self, other)
    }
}
