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
        var bb = entry
        var visited: ObjectSet<BasicBlock> = []
        var visitStack: [BasicBlock] = []
        var inStack: ObjectSet<BasicBlock> = []
        var result: [(BasicBlock, BasicBlock)] = []

        /// Initialization
        visited.insert(bb)
        visitStack.append(bb)
        inStack.insert(bb)

        repeat {
            let parent = visitStack.removeFirst()
            var foundNew = false
            for succ in parent.successors {
                bb = succ
                visited.insert(bb)
                if bb.hasSuccessors {
                    foundNew = true
                    break
                }
                /// Successor is in visitStack, it's a back edge
                if inStack.contains(bb) {
                    result.append((parent, bb))
                }
            }
            if foundNew {
                inStack.insert(bb)
                visitStack.append(bb)
            } else {
                inStack.remove(visitStack.last!)
            }
        } while !visitStack.isEmpty

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
