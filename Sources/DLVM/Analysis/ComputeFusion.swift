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

/// This file contains the Compute Fusion analysis for compute kernel
/// generation. Compute fusion performs BFS to fuse the following patterns
/// into one single kernel:
/// 1. Parallel element-wise expressions
/// 2. Sequential element-wise expressions

/// Symbolically represents a buffer for kernel computation
public struct ComputeBuffer {
    public var identifier: Int
    public var type: Type
}

/// Represents a node on either host or device
public enum DeviceSelectionNode {
    case host(ArraySlice<Instruction>)
    case device(ComputeNode)
}

/// Represents an argument (input) to the compute graph
public enum ComputeArgument {
    case instruction(Instruction)
    case argument(Argument)
}

/// A node containing instructions that are fused in one compute kernel
public class ComputeNode : BackwardGraphNode, HashableByReference {
    /// Parent basic block
    public unowned let parent: BasicBlock
    /// Predecessor compute nodes
    public fileprivate(set) var predecessors: ObjectSet<ComputeNode> = []
    /// Topo-sorted instructions
    public fileprivate(set) var instructions: ArraySlice<Instruction> = []
    /// List of kernel arguments
    public fileprivate(set) var arguments: [ComputeArgument] = []
    /// Buffers associated with each instruction to store the result
    public fileprivate(set) var buffers: [Instruction : ComputeBuffer] = [:]

    fileprivate init(parent: BasicBlock) {
        self.parent = parent
    }
}

public extension ComputeNode {
    /// Get the assigned buffer for instruction
    /// - Precondition: `inst` exists in the current compute node
    func buffer(for inst: Instruction) -> ComputeBuffer {
        guard let buffer = buffers[inst] else {
            preconditionFailure("Instruction does not exist in the current basic block")
        }
        return buffer
    }
}

open class ComputeFusionAnalysis : AnalysisPass {
    public typealias Body = BasicBlock

    private static func makeComputeBuffer(ofType type: Type, updating identifier: inout Int) -> ComputeBuffer {
        defer { identifier += 1 }
        return ComputeBuffer(identifier: identifier, type: type)
    }

    open class func run(on body: BasicBlock) throws -> DirectedGraph<ComputeNode> {
        var currentBufferID = 0
        var visited: Set<Instruction> = []
        let function = body.parent
        let users = try function.analysis(from: UserAnalysis.self)
        for inst in body {

        }
        DLUnimplemented()
    }
}
