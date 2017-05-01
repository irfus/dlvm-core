//
//  ComputeFusion.swift
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

import Foundation

/// A node containing all instructions that are fused in one kernel
/// Essentially a node in the meta-DFG
public struct ComputeNode : BackwardGraphNode {
    public var predecessors: [ComputeNode]
    public var instructions: [Instruction]
}

public class ComputeBasicBlock : HashableByReference {
    public var nodes: [ComputeNode] = []
}

open class ComputeFusion : AnalysisPass {
    public typealias Body = Function
    open class func run(on body: Function) -> DirectedGraph<ComputeBasicBlock> {
        DLUnimplemented()
    }
}
