//
//  DFG.swift
//  DLVM
//
//  Created by Richard Wei on 3/2/17.
//
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

public class DataFlowGraphAnalysis : AnalysisPass<Function, DirectedGraph<Instruction>> {
    public static override func run(on body: Function) throws -> DirectedGraph<Instruction> {
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
