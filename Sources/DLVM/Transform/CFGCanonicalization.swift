//
//  CFGCanonicalization.swift
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

/// Canonicalizes control flow graph so that it becomes "reversible": its
/// transpose is also a valid control flow graph.
/// Specifically:
/// - Merges multiple exits into a single exit.
/// - Adds "join" blocks so that each basic block has at most two predecessors.

open class CFGCanonicalization : TransformPass {
    public typealias Body = Function
    
    public static func run(on body: Function) -> Bool {
        var changed = false
        let builder = IRBuilder(function: body)
        var cfg = body.analysis(from: ControlFlowGraphAnalysis.self)

        /// Perform general CFG canonicalizations.
        /// Merge multiple exits.
        if body.premise.exits.count > 1 {
            mergeMultipleExits(in: body, using: builder, controlFlow: &cfg)
        }

        /// Form join blocks, if necessary.
        var postDomTrees = body.analysis(from: PostdominanceAnalysis.self)
        guard postDomTrees.count == 1 else { DLImpossible() }
        var visited: Set<BasicBlock> = []
        for bb in cfg.traversed(from: body[0], in: .breadthFirst) {
            bb.successors.lazy
                .filter { !visited.contains($0) }
                .forEach {
                    changed = formJoinBlock(
                        for: $0, predecessor: bb, using: builder,
                        postDominance: &postDomTrees) || changed
                }
            visited.insert(bb)
        }

        /// Canonicalize loops.
        var loopInfo = body.analysis(from: LoopAnalysis.self)
        for loop in loopInfo.uniqueLoops {
            /// If loop doesn't have a preheader, insert one.
            if loop.preheader == nil {
                insertPreheader(for: loop, loopInfo: &loopInfo, controlFlow: &cfg)
                changed = true
            }
            /// Next, check to make sure that all exit nodes of the loop only
            /// have predecessors that are inside of the loop. This check
            /// guarantees that the loop preheader/header will dominate the
            /// exit blocks. If the exit block has predecessors from outside of
            /// the loop, split the edge now.
            if !loop.hasDedicatedExits {
                changed = formDedicatedExits(
                    for: loop, loopInfo: &loopInfo, controlFlow: &cfg) || changed
            }
        }
        return changed
    }

    private static func mergeMultipleExits(
        in function: Function, using builder: IRBuilder,
        controlFlow cfg: inout ControlFlowGraphAnalysis.Result
    ) {
        let newExitArg = (function.makeFreshName("exit_value"), function.returnType)
        let newExit = BasicBlock(name: function.makeFreshBasicBlockName("exit"),
                                 arguments: [newExitArg],
                                 parent: function)
        for (exit, returnInst) in function.premise.exits {
            guard case let .return(use) = returnInst.kind else {
                fatalError("Invalid exit return instruction")
            }
            returnInst.kind = .branch(newExit, use.flatMap { [$0] } ?? [])
            cfg.insertEdge(from: exit, to: newExit)
        }
        function.append(newExit)
        builder.move(to: newExit)
        builder.return(%newExit.arguments[0])
    }

    private static func formJoinBlock(
        for bb: BasicBlock, predecessor: BasicBlock, using builder: IRBuilder,
        postDominance: inout PostdominanceAnalysis.Result
    ) -> Bool {
        var changed = false
        // If `bb` has less than two successors, return.
        if bb.successors.count < 2 { return false }
        /// Otherwise, check post-dominators.
        /// Precondition: `bb` must have at least 2 outgoing edges.
        let function = bb.parent
        for postDomTree in postDominance.values.filter({ $0.contains(bb) }) {
            let predDom = postDomTree.immediateDominator(of: predecessor)
            let bbDom = postDomTree.immediateDominator(of: bb)
            /// If post-dominators of `bb` and predecessor are the same, then
            /// create a join block.
            if predDom != bbDom { continue }
            let joinBlock = BasicBlock(
                name: function.makeFreshBasicBlockName("\(bb.name)_join"),
                arguments: predDom.arguments.map{(bb.makeFreshName($0.name), $0.type)},
                parent: function)
            function.insert(joinBlock, before: predDom)
            builder.move(to: joinBlock)
            builder.branch(predDom, joinBlock.arguments.map(%))
            /// Update branches.
            var visited: Set<BasicBlock> = []
            func redirectToJoinBlock(_ bb: BasicBlock) {
                if visited.contains(bb) || bb == joinBlock { return }
                visited.insert(bb)
                bb.premise.terminator.substituteBranches(to: bbDom, with: joinBlock)
                bb.successors.forEach(redirectToJoinBlock)
            }
            redirectToJoinBlock(bb)
            changed = true
        }
        return changed
    }

    private static func insertPreheader(
        for loop: Loop, loopInfo: inout LoopInfo,
        controlFlow cfg: inout DirectedGraph<BasicBlock>
    ) {
        /// Gather original predecessors of header.
        let preds = cfg.predecessors(of: loop.header)
            .lazy.filter { !loop.contains($0) }
        /// Create preheader and hoist predecessors to it.
        let preheader = loop.header.hoistPredecessorsToNewBlock(
            named: "preheader", hoisting: preds, controlFlow: &cfg)
        /// Make the preheader a part of the parent loop if it exists.
        if let parent = loop.parent {
            loopInfo.innerMostLoops[preheader] = parent
            parent.blocks.insert(preheader, before: loop.header)
        }
    }

    private static func formDedicatedExits(
        for loop: Loop, loopInfo: inout LoopInfo,
        controlFlow cfg: inout DirectedGraph<BasicBlock>
    ) -> Bool {
        var changed = false
        for exit in loop.exits {
            /// Gather predecessors that are inside loop.
            let preds = cfg.predecessors(of: exit)
            let insidePreds = preds.lazy.filter { loop.contains($0) }
            guard insidePreds.count < preds.count else { continue }
            /// Create new exit and hoist inside-predecessors to it.
            let newExit = exit.hoistPredecessorsToNewBlock(
                named: "exit", hoisting: insidePreds, controlFlow: &cfg)
            /// Make the new exit a part of the parent loop if it exists.
            if let parent = loop.parent {
                loopInfo.innerMostLoops[newExit] = parent
                parent.blocks.insert(newExit, before: exit)
            }
            changed = true
        }
        return changed
    }
}
