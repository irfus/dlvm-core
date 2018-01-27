//
//  Dominance.swift
//  DLVM
//
//  Copyright 2016-2018 The DLVM Team.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

/// This file contains an implementation of the Cooper-Harvey-Kennedy dominance
/// algorithm.
/// K. D. Cooper, T. J. Harvey, and K. Kennedy. A simple, fast dominance
/// algorithm. Rice University, 2000.


public struct DominatorTree<Node : IRUnit> {
    public unowned let root: Node
    fileprivate var graph = DirectedGraph<Node>()
}

extension DominatorTree : BidirectionalEdgeSet {
    public func predecessors(of node: Node) -> OrderedSet<Node> {
        return graph.predecessors(of: node)
    }

    public func successors(of node: Node) -> OrderedSet<Node> {
        return graph.successors(of: node)
    }
}

public extension DominatorTree {
    init(root: Node) {
        self.root = root
        self.graph.insertEdge(from: root, to: root)
    }

    mutating func updateImmediateDominator(_ dominator: Node, for node: Node) {
        if contains(node), let oldDom = graph.predecessors(of: node).first {
            graph.removeEdge(from: oldDom, to: node)
        }
        graph.insertEdge(from: dominator, to: node)
    }

    func immediateDominator(of node: Node) -> Node {
        guard let idom = predecessors(of: node).first else {
            fatalError("\(node) does not have an immediate dominator")
        }
        return idom
    }

    func immediateDominatees(of node: Node) -> OrderedSet<Node> {
        return successors(of: node)
    }

    func nearestCommonDominator(_ b1: Node, _ b2: Node) -> Node {
        if dominates(b1, b2) { return b1 }
        if dominates(b2, b1) { return b2 }

        /// Collect all b1's dominators
        var b1Dominators: Set<Node> = []
        b1Dominators.insert(root)
        var b1Dom = immediateDominator(of: b1)
        while b1Dom !== root {
            b1Dominators.insert(b1Dom)
            b1Dom = immediateDominator(of: b1Dom)
        }

        /// Check all b2's dominators against b1's dominators
        var b2Dom = immediateDominator(of: b2)
        while b2Dom !== root {
            if b1Dominators.contains(b2Dom) {
                return b2Dom
            }
            b2Dom = immediateDominator(of: b2Dom)
        }

        return root
    }

    func contains(_ node: Node) -> Bool {
        return graph.contains(node)
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
    func properlyDominates(_ instruction: Instruction,
                           _ otherInstruction: Instruction) -> Bool {
        let bb1 = instruction.parent, bb2 = otherInstruction.parent
        if bb1 !== bb2 {
            return properlyDominates(bb1, bb2)
        }
        return instruction.indexInParent < otherInstruction.indexInParent
    }
}

public extension DominatorTree where Node == BasicBlock {
    func properlyDominates(_ use: Use, _ instruction: Instruction) -> Bool {
        switch use {
        case .variable, .literal, .function: return true

        case let .argument(arg):
            return self.dominates(arg.parent, instruction.parent)

        case let .instruction(usee):
            return properlyDominates(usee, instruction)
        }
    }
}

public extension IRUnit {
    func dominates(_ other: Self,
                   in domTree: DominatorTree<Self>) -> Bool {
        return domTree.dominates(self, other)
    }

    func properlyDominates(_ other: Self,
                           in domTree: DominatorTree<Self>) -> Bool {
        return domTree.properlyDominates(self, other)
    }
}

public extension Instruction {
    func properlyDominates(_ other: Instruction,
                           in domTree: DominatorTree<BasicBlock>) -> Bool {
        return domTree.properlyDominates(self, other)
    }
}

open class DominanceAnalysis : AnalysisPass {
    public typealias Body = Function
    public typealias Result = DominatorTree<BasicBlock>

    open class func run(on body: Function) -> DominatorTree<BasicBlock> {
        let entry = body.premise.entry
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
                    .lazy.filter(domTree.contains)
                guard var newIDom = preds.first else {
                    preconditionFailure("""
                        Successor node doesn't have any predecessor
                        """)
                }
                for p in preds.dropFirst() {
                    newIDom = domTree.nearestCommonDominator(p, newIDom)
                }
                if !domTree.contains(node) ||
                    domTree.immediateDominator(of: node) !== newIDom {
                    domTree.updateImmediateDominator(newIDom, for: node)
                    changed = true
                }
            }
        } while changed
        return domTree
    }
}

/// Post-dominance analysis on a function
open class PostdominanceAnalysis : AnalysisPass {
    public typealias Body = Function
    public typealias Result = [BasicBlock : DominatorTree<BasicBlock>]

    open class func run(on body: Function) -> Result {
        let exits = body.premise.exits
        let cfg = body.analysis(from: ControlFlowGraphAnalysis.self)
        let transposeCFG = cfg.transpose
        var domTrees: Result = [:]
        for (exit, _) in exits {
            var domTree = DominatorTree(root: exit)
            /// Iteratively build tree
            var changed = true
            repeat {
                changed = false
                let rpo = transposeCFG.traversed(from: exit,
                                                 in: .postorder).reversed()
                for node in rpo.dropFirst() {
                    let preds = transposeCFG.predecessors(of: node)
                        .lazy.filter(domTree.contains)
                    guard var newIDom = preds.first else {
                        preconditionFailure("""
                            Successor node doesn't have any predecessor
                            """)
                    }
                    for p in preds.dropFirst() {
                        newIDom = domTree.nearestCommonDominator(p, newIDom)
                    }
                    if !domTree.contains(node) ||
                        domTree.immediateDominator(of: node) !== newIDom {
                        domTree.updateImmediateDominator(newIDom, for: node)
                        changed = true
                    }
                }
            } while changed
            domTrees[exit] = domTree
        }
        return domTrees
    }
}
