//
//  DFG.swift
//  DLVM
//
//  Created by Richard Wei on 3/2/17.
//
//

open class ControlFlowGraphAnalysis : AnalysisPass<Function, DirectedGraph<BasicBlock>> {
    open override class func run(on body: Function) -> DirectedGraph<BasicBlock> {
        return DirectedGraph(nodes: body)
    }
}
