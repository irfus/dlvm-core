//
//  DFG.swift
//  DLVM
//
//  Created by Richard Wei on 3/2/17.
//
//

open class SectionGraphAnalysis : AnalysisPass<Function, DirectedGraph<Section>> {
    open override class func run(on body: Function) -> DirectedGraph<Section> {
        return DirectedGraph(nodes: body)
    }
}

open class ControlFlowGraphAnalysis : AnalysisPass<Section, DirectedGraph<BasicBlock>> {
    open override class func run(on body: Section) -> DirectedGraph<BasicBlock> {
        return DirectedGraph(nodes: body)
    }
}
