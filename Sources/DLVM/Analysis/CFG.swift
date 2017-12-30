//
//  CFG.swift
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

extension BasicBlock : ForwardGraphNode {
    public var successors: ObjectSet<BasicBlock> {
        return terminator?.controlFlowSuccessors ?? []
    }
}

public extension Instruction {

    var controlFlowSuccessors: ObjectSet<BasicBlock> {
        switch kind {
        case let .branch(bb, _):
            return [bb]
        case let .conditional(_, bb1, _, bb2, _):
            return [bb1, bb2]
        default: return []
        }
    }

    var controlFlowSuccessorCount: Int {
        switch kind {
        case .branch: return 1
        case .conditional: return 2
        default: return 0
        }
    }


    /// Return true if the specified edge is a critical edge.
    /// Critical edges are edges from a block with multiple successors to a block
    /// with multiple predecessors.
    func isCriticalEdge(to destination: BasicBlock) throws -> Bool {
        let function = destination.parent
        precondition(function === parent.parent, """
                     Destination basic block is not in the same function as
                     the instruction
                     """)
        let cfg = function.analysis(from: ControlFlowGraphAnalysis.self)
        if controlFlowSuccessorCount <= 1 { return false }
        return cfg[destination].predecessors.count > 1
    }

}

public extension Function {

    /// Compute and returns back edges in function
    func backEdges(fromEntry entry: BasicBlock) -> [(BasicBlock, BasicBlock)] {
        var visited: ObjectSet<BasicBlock> = []
        var inStack: ObjectSet<BasicBlock> = []
        var result: [(BasicBlock, BasicBlock)] = []

        /// Define helper function
        func backEdgesHelper(_ bb: BasicBlock) {
            visited.insert(bb)
            inStack.insert(bb)

            for succ in bb.successors {
                if !visited.contains(succ) {
                    backEdgesHelper(succ)
                } else if inStack.contains(succ) {
                    result.append((bb, succ))
                }
            }

            inStack.remove(bb)
        }

        /// Call helper function on entry
        backEdgesHelper(entry)
        return result
    }

}

open class ControlFlowGraphAnalysis : AnalysisPass {
    public typealias Body = Function
    public typealias Result = DirectedGraph<BasicBlock>

    open class func run(on body: Function) -> DirectedGraph<BasicBlock> {
        return DirectedGraph<BasicBlock>(nodes: body)
    }
}
