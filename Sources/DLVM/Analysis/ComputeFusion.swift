//
//  ComputeFusion.swift
//  DLVM
//
//  Created by Richard Wei on 3/26/17.
//
//

import Foundation

/// A node containing all instructions that are fused in one kernel
public struct ComputeNode {
    /// TODO
}

/// A basic block containing compute nodes
public struct ComputeBasicBlock {
    /// TODO
}

public class ComputeFusion : AnalysisPass<BasicBlock, ComputeBasicBlock> {
    public static override func run(on body: BasicBlock) -> ComputeBasicBlock {
        fatalError("Unimplemented")
    }
}
