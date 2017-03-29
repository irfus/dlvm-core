//
//  ComputeFusion.swift
//  DLVM
//
//  Created by Richard Wei on 3/26/17.
//
//

import Foundation

/// A node containing all instructions that are fused in one kernel
/// Essentially a node in the meta-DFG
public struct ComputeNode : BackwardGraphNode {
    public var predecessors: [ComputeNode]
    public var instructions: [Instruction]
}

public class ComputeBasicBlock {
    public var nodes: [ComputeNode] = []
}

public class ComputeFusion : AnalysisPass<Function, DirectedGraph<ComputeBasicBlock>> {
    public static override func run(on body: Function) -> DirectedGraph<ComputeBasicBlock> {
        return DLUnimplemented()
    }
}
