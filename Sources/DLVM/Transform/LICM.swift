//
//  LICM.swift
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

/// Loop invariant code motion, based on SIL's implementation:
/// https://github.com/apple/swift/blob/master/lib/SILOptimizer/LoopTransforms/LICM.cpp
open class LoopInvariantCodeMotion : TransformPass {
    public typealias Body = Function

    public static func run(on function: Function) -> Bool {
        var changed = false
        let loopInfo = function.analysis(from: LoopAnalysis.self)
        var domTree = function.analysis(from: DominanceAnalysis.self)
        let sideEff = function.parent.analysis(from: SideEffectAnalysis.self)
        if loopInfo.isEmpty {
            return false
        }
        for tll in loopInfo.topLevelLoops {
            // FIXME: Implement
            let lto = LoopTreeOptimization(
                topLevelLoop: tll, loopInfo: loopInfo,
                sideEffectInfo: sideEff,
                dominatorTree: domTree)
            changed = lto.optimize() || changed
            // If changed, recompute dominator tree
            if changed {
                domTree = function.analysis(from: DominanceAnalysis.self)
            }
        }
        return changed
    }
}

/// Summary of may-writes occurring in the loop tree rooted at `loop`. This
/// includes all writes of the sub-loops and the loop itself.
private struct LoopNestSummary {
    var loop: Loop
    var possibleWrites: Set<Instruction> = []
}

private final class LoopTreeOptimization {
    var loopNestSummaryMap: [Loop : LoopNestSummary] = [:]
    var bottomUpWorkList: [Loop] = []
    let loopInfo: LoopInfo
    var dominatorTree: DominatorTree<BasicBlock>
    var sideEffectInfo: SideEffectInfo
    
    init(topLevelLoop: Loop, loopInfo: LoopInfo,
         sideEffectInfo: SideEffectInfo,
         dominatorTree: DominatorTree<BasicBlock>) {
        self.loopInfo = loopInfo
        self.sideEffectInfo = sideEffectInfo
        self.dominatorTree = dominatorTree
        // Collect loops for a recursive bottom-up traversal in the loop tree.
        func collectAll(_ loop: Loop) {
            bottomUpWorkList.append(loop)
            for subloop in loop.subloops {
                collectAll(subloop)
            }
        }
        collectAll(topLevelLoop)
    }
}

extension LoopTreeOptimization {
    func optimize() -> Bool {
        // Return true if changed.
        // FIXME: Implement
        return false
    }
}
