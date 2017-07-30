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

public enum ComputationKind {
    /// Data transfer not needed
    case hostNoTransfer
    /// Data transfer needed
    case hostSequential
    case hostParallel
    case hostBLAS(BLAS)
    case kernel
    case deviceBLAS(BLAS)
}

extension ComputationKind {
    var isHost: Bool {
        switch self {
        case .hostBLAS(_), .hostParallel, .hostSequential:
            return true
        default:
            return false
        }
    }

    var isDevice: Bool {
        return !isHost
    }
    
    var fusable: Bool {
        switch self {
        case .deviceBLAS(_), .hostBLAS(_):
            return false
        default:
            return true
        }
    }
}

/// A subgraph of DFG (basic block)
public class FusionDataFlowNode<Target : ComputeTarget>
    : BackwardGraphNode, HashableByReference {
    /// Node kind
    public let computationKind: ComputationKind
    /// Parent basic block
    public unowned let parent: BasicBlock
    /// Topo-sorted instructions
    public fileprivate(set) var instructions: OrderedSet<Instruction> = []
    /// Predecessors preserving information about execution order
    public fileprivate(set) var predecessors: ObjectSet<FusionDataFlowNode> = []

    fileprivate init(computationKind: ComputationKind, parent: BasicBlock) {
        self.computationKind = computationKind
        self.parent = parent
    }
}

extension Instruction {

    enum DefaultExecutionLocation {
        case host
        case device
    }
    
    var defaultExecutionLocation: DefaultExecutionLocation {
        switch kind.opcode {
        case .unaryOp, .binaryOp, .dataTypeCast, .transpose:
            guard case let .tensor(shape, _) = type else {
                DLImpossible()
            }
            let size = shape.contiguousSize
            return size == 1 ? .host : .device
        case .bitCast, .shapeCast, .extract, .elementPointer, .branch,
             .apply, .return, .trap, .allocateBox, .allocateHeap,
             .allocateStack, .concatenate, .copy, .conditional, .deallocate,
             .insert, .load, .projectBox, .release, .retain, .scan, .reduce,
             .slice, .store:
            return .host
        case .matrixMultiply:
            return .device
        }
    }

    var needsToRequestMemory: Bool {
        switch kind.opcode {
        case .unaryOp, .binaryOp, .dataTypeCast:
            guard case let .tensor(shape, _) = type else {
                DLImpossible()
            }
            let size = shape.contiguousSize
            return size != 1
        case .bitCast, .shapeCast, .extract,
             .elementPointer, .branch, .apply, .return, .trap:
            return false
        case .allocateBox, .allocateHeap, .allocateStack, .concatenate, .copy,
             .conditional, .deallocate, .insert, .load, .projectBox, .release,
             .retain, .scan, .reduce, .slice, .store, .matrixMultiply,
             .transpose:
            return true
        }
    }
}

class ComputeFusionAnalysis<Target : ComputeTarget> : AnalysisPass {
    typealias Body = BasicBlock
    typealias Node = FusionDataFlowNode<Target>
    typealias Result = [Node]

    private static func visit(from instructions: [Instruction],
                              basicBlock bb: BasicBlock,
                              parentSubgraphs: inout [Instruction : Node],
                              dataFlow dfg: DataFlowGraph,
                              visited: inout Set<Instruction>) -> Node {
//        let bfs = dfg.breadthFirst(from: instructions as [Definition])
//        for (depth, inst) in bfs {
//
//        }
        DLUnimplemented()
    }

    class func run(on body: BasicBlock) -> [Node] {
        let fn = body.parent
        var visited: Set<Instruction> = []
        var parentSubgraphs: [Instruction : Node] = [:]
        var nodes: [Node] = []
        /// - TODO:
        /// BFS from unvisited instructions and collect subgraphs
        let dfg = fn.analysis(from: DataFlowGraphAnalysis.self)
        /// Visit device functions
        let entry = body.premise.first
        let node = visit(from: [entry], basicBlock: body,
                         parentSubgraphs: &parentSubgraphs,
                         dataFlow: dfg, visited: &visited)
        nodes.append(node)
        /// Night!
        return nodes
    }
}
