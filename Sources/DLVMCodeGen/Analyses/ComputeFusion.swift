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

import DLVM

/// This file contains the Compute Fusion analysis for compute kernel
/// generation. Compute fusion performs BFS to fuse the following patterns
/// into one single kernel:
/// 1. Parallel element-wise expressions
/// 2. Sequential element-wise expressions

/// A subgraph of DFG (basic block)
public class FusionDataFlowNode<Target : ComputeTarget> : BackwardGraphNode, HashableByReference {
    public enum Kind {
        case host
        case hostIndirect
        case kernel
        case deviceBLAS(BLAS)
    }
    /// Node kind
    public let kind: Kind
    /// Parent basic block
    public unowned let parent: BasicBlock
    /// Topo-sorted instructions
    public fileprivate(set) var instructions: OrderedSet<Instruction> = []
    /// Predecessors preserving information about execution order
    public fileprivate(set) var predecessors: ObjectSet<FusionDataFlowNode> = []

    fileprivate init(kind: Kind, parent: BasicBlock) {
        self.kind = kind
        self.parent = parent
    }
}

class ComputeFusionAnalysis<Target : ComputeTarget> : AnalysisPass {
    typealias Body = BasicBlock
    typealias Node = FusionDataFlowNode<Target>

    private static func visit(from instruction: Instruction,
                              parentSubgraphs: inout [Instruction : Node]) throws -> Node {
        let bb = instruction.parent
        let fn = bb.parent
        let users = try fn.analysis(from: UserAnalysis.self)
        var queue: ArraySlice<Instruction?> = [instruction]
        var level = 0
        /// Check for subroutine pattern
        if let blasCapableTarget = Target.self as? BLASCapable.Type,
            let (blas, insts) = blasCapableTarget.blasFusion(from: instruction) {
            let subgraph = Node(kind: .deviceBLAS(blas), parent: bb)
            return subgraph
        }
        /// If `instruction` is a non-elementwise barrier, it's a single node
        switch instruction.kind {
        case .matrixMultiply, .transpose:
            DLUnimplemented()
        default:
            DLUnimplemented()
        }
        DLUnimplemented()
        /// BFS from `instruction` to collect nodes into the subgraph
        repeat {
            switch queue.removeFirst() {
            case nil:
                level += 1
            case let inst?:
                /// - TODO: Visit instruction
                /// Fusion conditions:
                /// - Whatever host-only subgraph
                /// - Device data flow of level <= 5 such that
                ///   - Only leaf nodes have external users
                ///   - Non-elementwise instructions are barriers
                /// Push users
                for user in users[inst, bb] {
                    queue.append(user)
                }
                /// Level up
                queue.append(nil)
            }
        } while !queue.isEmpty
    }

    class func run(on body: BasicBlock) throws -> [Node] {
        var visited: Set<Instruction> = []
        var parentSubgraphs: [Instruction : Node] = [:]
        var nodes: [Node] = []
        for inst in body where !visited.contains(inst) {
            let node = try visit(from: inst, parentSubgraphs: &parentSubgraphs)
            node.instructions.forEach { visited.insert($0) }
            nodes.append(node)
        }
        return nodes
    }
}
