//
//  LoopAnalysis.swift
//  DLVM
//
//  Copyright 2016-2017 The DLVM Team.
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

/// This file defines the LoopInfo class that is used to identify natural loops
/// and determine the loop depth of various nodes of the CFG. A natural loop
/// has exactly one entry-point, which is called the header. Note that natural
/// loops may actually be several loops that share the same header node.
///
/// This analysis calculates the nesting structure of loops in a function. For
/// each natural loop identified, this analysis identifies natural loops
/// contained entirely within the loop and the basic blocks that make up the
/// loop.

public class Loop : EquatableByReference {
    public fileprivate(set) weak var parent: Loop?
    public fileprivate(set) var subloops: [Loop] = []
    public fileprivate(set) var blocks: OrderedSet<BasicBlock>

    public init(parent: Loop? = nil, header: BasicBlock) {
        self.parent = parent
        self.blocks = [header]
    }
}

public extension Loop {
    var header: BasicBlock {
        guard let first = blocks.first else { DLImpossible() }
        return first
    }
    
    var function: Function {
        return header.parent
    }
    
    /// Return all blocks inside the loop that have successors outside of the
    /// loop. These are the blocks _inside of the current loop_ which branch out.
    /// The returned list is always unique.
    var exitingBlocks: [BasicBlock] {
        var exiting: [BasicBlock] = []
        for bb in blocks {
            for succ in bb.successors where !contains(succ) {
                exiting.append(bb)
            }
        }
        return exiting
    }
    
    /// Return all of the successor blocks of this loop. These are the blocks
    /// _outside of the current loop_ which are branched to.
    var exits: [BasicBlock] {
        return blocks.lazy.flatMap{$0.successors}.filter{!contains($0)}
    }
    
    var exitEdges: [(source: BasicBlock, destination: BasicBlock)] {
        return blocks.lazy
            .flatMap { bb in bb.successors.map { (bb, $0) } }
            .filter { !contains($0.1) }
    }
    
    /// If the given loop's header has exactly one unique predecessor outside
    /// the loop, return it. Otherwise return null. This is less strict that the
    /// loop "preheader" concept, which requires the predecessor to have exactly
    /// one successor.
    var predecessor: BasicBlock? {
        var out: BasicBlock?
        let cfg = function.analysis(from: ControlFlowGraphAnalysis.self)
        for pred in cfg.predecessors(of: header) where !contains(pred) {
            if let out = out, out != pred { return nil }
            out = pred
        }
        precondition(out != nil, "Header of loop has no predecessors from outside loop")
        return out
    }

    /// If there is a single latch block for this loop, return it. A latch block
    /// is a block that contains a branch back to the header.
    var latch: BasicBlock? {
        let cfg = function.analysis(from: ControlFlowGraphAnalysis.self)
        var latch: BasicBlock?
        for pred in cfg.predecessors(of: header) where contains(pred) {
            if latch != nil { return nil }
            latch = pred
        }
        return latch
    }
}

public extension Loop {
    func contains(_ block: BasicBlock) -> Bool {
        return blocks.contains(block)
    }
    
    func contains(_ inst: Instruction) -> Bool {
        return contains(inst.parent)
    }

    func isLoopInvariant(_ inst: Instruction) -> Bool {
        return !contains(inst)
    }
    
    func isLoopInvariant(_ v: Value) -> Bool {
        return (v as? Instruction).map { !contains($0) } ?? true
    }
    
    func hasLoopInvariantOperands(_ inst: Instruction) -> Bool {
        return inst.operands.forAll { isLoopInvariant($0.value) }
    }
    
    var hasDedicatedExits: Bool {
        let cfg = function.analysis(from: ControlFlowGraphAnalysis.self)
        // Each predecessor of each exit block of a normal loop is contained
        // within the loop.
        for bb in exits {
            for pred in cfg.predecessors(of: bb) where !contains(pred) {
                return false
            }
        }
        return !exits.lazy.flatMap(cfg.predecessors).contains { !self.contains($0) }
    }
    
    var uniqueExits: [BasicBlock] {
        DLUnimplemented()
    }
    
    var canonicalInductionVariable: Set<Argument> {
        DLUnimplemented()
    }
}

public struct LoopInfo {
    public fileprivate(set) var innerMostLoops: [BasicBlock : Loop] = [:]
    public fileprivate(set) var topLevelLoops: [Loop] = []
}

/// Analyze LoopInfo discovers loops during a postorder DominatorTree traversal
/// interleaved with backward CFG traversals within each subloop. The backward
/// traversal skips inner subloops, so this part of the algorithm is linear in
/// the number of CFG edges. `subloops` and `blocks` in Loop are then populated
/// during a single forward CFG traversal.
///
/// During the two CFG traversals each block is seen three times:
/// 1) Discovered and mapped by a reverse CFG traversal.
/// 2) Visited during a forward DFS CFG traversal.
/// 3) Reverse-inserted in the loop in postorder following forward DFS.
///
/// The `blocks` sets are inclusive, so step 3 requires loop-depth number of
/// insertions per block.
open class LoopAnalysis : AnalysisPass {
    public typealias Body = Function
    public typealias Result = LoopInfo
    
    public static func run(on body: Function) -> LoopInfo {
        var info = LoopInfo()
        let cfg = body.analysis(from: ControlFlowGraphAnalysis.self)
        let domTree = body.analysis(from: DominanceAnalysis.self)

        for header in domTree.traversed(from: domTree.root, in: .postorder) {
            var backEdges: [BasicBlock] = []
            /// Check each predecessor of the potential loop header.
            for pred in cfg.predecessors(of: header) {
                if domTree.contains(pred) && header.dominates(pred, in: domTree) {
                    backEdges.append(pred)
                }
            }
            if !backEdges.isEmpty {
                let loop = Loop(header: header)
                /// Discover a subloop with the specified backedges such that: All blocks within
                /// this loop are mapped to this loop or a subloop. And all subloops within this
                /// loop have their parent loop set to this loop or a subloop.
                discoverAndMapSubloop(for: loop, backEdges: backEdges, info: &info, controlFlow: cfg, dominance: domTree)
            }
        }
        /// Perform a single forward CFG traversal to populate block and subloop vectors for all loops.
        populateLoops(from: domTree.root, into: &info)
        return info
    }

    private static func discoverAndMapSubloop(
        for loop: Loop, backEdges: [BasicBlock], info: inout LoopInfo,
        controlFlow cfg: ControlFlowGraphAnalysis.Result, dominance domTree: DominatorTree<BasicBlock>)
    {
        /// Perform a backward CFG traversal using a worklist.
        var workList = backEdges
        while !workList.isEmpty {
            let block = workList.removeLast()
            guard var subloop = info.innerMostLoops[block] else {
                if !domTree.contains(block) {
                    continue
                }
                /// This is an undiscovered block. Map it to the current loop.
                info.innerMostLoops[block] = loop
                if block == loop.header {
                    continue
                }
                /// Push all block predecessors on the worklist.
                workList.append(contentsOf: cfg.predecessors(of: block))
                continue
            }
            /// This is a discovered block. Find its outermost discovered loop.
            while let parent = subloop.parent {
                subloop = parent
            }
            /// If it is already discovered to be a subloop of this loop, continue.
            if subloop == loop {
                continue
            }
            /// Discover a subloop of this loop.
            subloop.parent = loop
            /// Continue traversal along predecessors that are not loop-back edges from
            /// within this subloop tree itself. Note that a predecessor may directly
            /// reach another subloop that is not yet discovered to be a subloop of
            /// this loop, which we must traverse.
            for pred in cfg.predecessors(of: block) {
                if info.innerMostLoops[pred] != subloop {
                    workList.append(pred)
                }
            }
        }
    }

    private static func populateLoops(from header: BasicBlock, into info: inout LoopInfo) {
        for bb in header.postorder {
            /// Add a single block to its ancestor loops in postorder. If the
            /// block is a subloop header, add the subloop to its parent in
            /// postorder, then reverse the block and subloop arrays of the
            /// now complete subloop to achieve RPO.
            guard var subloop = info.innerMostLoops[bb] else { continue }
            if bb == subloop.header {
                /// We reach this point once per subloop after processing all
                /// the blocks in the subloop.
                if let parent = subloop.parent {
                    parent.subloops.append(subloop)
                } else {
                    info.topLevelLoops.append(subloop)
                }
                /// For convenience, blocks and subloops are inserted in postorder.
                /// Reverse the lists, except for the loop header, which is always
                /// at the beginning.
                subloop.blocks[1...].reverse()
                subloop.subloops.reverse()
                guard let parent = subloop.parent else {
                    continue
                }
                subloop = parent
            }
            subloop.blocks.append(bb)
            while let parent = subloop.parent {
                parent.blocks.append(bb)
                subloop = parent
            }
        }
    }
}
