//
//  DFG.swift
//  DLVM
//
//  Copyright 2016-2017 Richard Wei.
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

extension Instruction : BackwardGraphNode {
    public var predecessors: ObjectSet<Instruction> {
        var preds: ObjectSet<Instruction> = []
        for case .instruction(_, let inst) in operands {
            preds.insert(inst)
        }
        return preds
    }
}

open class DataFlowGraphAnalysis : AnalysisPass<Function, DirectedGraph<Instruction>> {
    open override class func run(on body: Function) throws -> DirectedGraph<Instruction> {
        var graph = DirectedGraph<Instruction>()
        for bb: BasicBlock in body {
            for inst: Instruction in bb {
                for case .instruction(_, let usedInst) in inst.operands {
                    graph.insertEdge(from: usedInst, to: inst)
                }
            }
        }
        return graph
    }
}
