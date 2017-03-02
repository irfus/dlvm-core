//
//  DominanceAnalysis.swift
//  DLVM
//
//  Created by Richard Wei on 2/18/17.
//

open class DominanceAnalysis : AnalysisPass<Function, DominatorTree?> {
    open override func run(on body: Function) -> DominatorTree? {
        return nil
    }
}

open class PostdominanceAnalysis : AnalysisPass<Function, DominatorTree?> {
    open override func run(on body: Function) -> DominatorTree? {
        return nil
    }
}
