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


public struct DominatorTree<Node : IRUnit> {
    fileprivate var immediateDominators: [Node : Node] = [:]
    public unowned let root: Node
}

public extension DominatorTree {

    public init(root: Node) {
        self.root = root
        self.immediateDominators[root] = root
    }

    public mutating func updateImmediateDominator(_ dominator: Node, for node: Node) {
        immediateDominators[node] = dominator
    }

}

public extension DominatorTree {

    public func immediateDominator(of node: Node) -> Node {
        return immediateDominators[node]!
    }

    public func nearestCommonDominator(_ b1: Node, _ b2: Node) -> Node {
        if dominates(b1, b2) { return b1 }
        if dominates(b2, b1) { return b2 }

        /// Collect all b1's dominators
        var b1Dominators: ObjectSet<Node> = []
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

    public func contains(_ node: Node) -> Bool {
        return immediateDominators.keys.contains(node)
    }

    func properlyDominates(_ block: Node, _ otherBlock: Node) -> Bool {
        guard block !== otherBlock else { return false }
        if block === root { return true }
        var iDom = immediateDominator(of: otherBlock)
        while iDom !== block && iDom !== root {
            iDom = immediateDominator(of: iDom)
        }
        return iDom !== root
    }

    func dominates(_ block: Node, _ otherBlock: Node) -> Bool {
        return block === otherBlock || properlyDominates(block, otherBlock)
    }
}

public extension DominatorTree where Node == BasicBlock {
    func immediateDominatorInstruction(of body: Node) -> Instruction? {
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

public extension IRUnit {
    public func isReachable(in domTree: DominatorTree<Self>) -> Bool {
        return domTree.contains(self)
    }

    public func dominates(_ other: Self,
                          in domTree: DominatorTree<Self>) -> Bool {
        return domTree.dominates(self, other)
    }
}

public extension Instruction {
    public func dominates(_ other: Instruction,
                          in domTree: DominatorTree<BasicBlock>) -> Bool {
        return domTree.dominates(self, other)
    }
}


open class DominanceAnalysis : AnalysisPass<Section, DominatorTree<BasicBlock>> {

    open override class func run(on body: Section) -> DominatorTree<BasicBlock> {
        let entry = body.entry!
        var domTree = DominatorTree(root: entry)
        let cfg = body.analysis(from: ControlFlowGraphAnalysis.self)

        /// Initialization of data flow analysis
        /// (done by the dom tree initializer)

        /// Iteratively build tree
        var changed = true
        repeat {
            changed = false
            for node in entry.postorder.reversed().dropFirst() {
                let preds = cfg.predecessors(of: node)
                guard var newIDom = preds.first else {
                    preconditionFailure("Successor node doesn't have any predecessor")
                }
                for p in preds.dropFirst() where !domTree.contains(p) {
                    newIDom = domTree.nearestCommonDominator(p, newIDom)
                }
                if domTree.immediateDominator(of: node) !== newIDom {
                    domTree.updateImmediateDominator(newIDom, for: node)
                    changed = true
                }
            }
        } while changed
        return domTree
    }
}

open class PostdominanceAnalysis : AnalysisPass<Section, [DominatorTree<BasicBlock>]> {

    static func leaves(in section: Section) -> [BasicBlock] {
        return section.filter { $0.isLeaf }
    }

    static func postdominatorTree(from leaf: BasicBlock,
                                  controlFlowGraph cfg: DirectedGraph<BasicBlock>) -> DominatorTree<BasicBlock> {
        var domTree = DominatorTree(root: leaf)
        let transposeCFG = cfg.transpose

        /// Initialization of data flow analysis
        /// (done by the dom tree initializer)

        /// Iteratively build tree
        var changed = true
        repeat {
            changed = false
            /// TODO: Need reverse postorder
            for node in transposeCFG.traversed(from: leaf, in: .postorder).reversed().dropFirst() {
                let preds = transposeCFG.predecessors(of: node)
                guard var newIDom = preds.first else {
                    preconditionFailure("Successor node doesn't have any predecessor")
                }
                for p in preds.dropFirst() where !domTree.contains(p) {
                    newIDom = domTree.nearestCommonDominator(p, newIDom)
                }
                if domTree.immediateDominator(of: node) !== newIDom {
                    domTree.updateImmediateDominator(newIDom, for: node)
                    changed = true
                }
            }
        } while changed
        return domTree
    }

    open override class func run(on body: Section) -> [DominatorTree<BasicBlock>] {
        let cfg = body.analysis(from: ControlFlowGraphAnalysis.self)
        return leaves(in: body).map { postdominatorTree(from: $0, controlFlowGraph: cfg) }
    }
}
