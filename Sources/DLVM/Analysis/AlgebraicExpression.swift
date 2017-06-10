//
//  AlgebraicExpression.swift
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

/// AlgebraicExpression is designed to simplify pattern matching for math
/// expressions in domain-specific optimizations such as Algebra Simplification,
/// Linear Algebra Fusion and Matrix Chain Ordering

public indirect enum AlgebraicExpression {
    case atom(Use)
    case unary(UnaryOp, AlgebraicExpression, Instruction)
    case binary(BinaryOp, AlgebraicExpression, AlgebraicExpression, Instruction)
    case matrixMultiply(AlgebraicExpression, AlgebraicExpression, Instruction)
    case transpose(AlgebraicExpression, Instruction)
}

extension AlgebraicExpression {
    /// Top instruction (post-dominator) in the expression
    var topInstruction: Instruction? {
        switch self {
        case .atom(_):
            return nil
        case .binary(_, _, _, let inst),
             .matrixMultiply(_, _, let inst),
             .transpose(_, let inst),
             .unary(_, _, let inst):
            return inst
        }
    }

    /// Remove intermediate instructions from the basic block
    func removeIntermediates() {
        switch self {
        case .atom(_):
            return
        case let .binary(_, lhs, rhs, inst),
             let .matrixMultiply(lhs, rhs, inst):
            lhs.removeIntermediates()
            rhs.removeIntermediates()
            inst.removeFromParent()
        case let .transpose(x, inst),
             let .unary(_, x, inst):
            x.removeIntermediates()
            inst.removeFromParent()
        }
    }

    func replaceExpression(with newInstruction: Instruction) {
        /// DFS-remove intermediate instructions
        guard let inst = topInstruction else { return }
        let bb = inst.parent
        let instIndex = inst.indexInParent
        /// Insert new instruction at the location of the top instruction
        bb.insert(newInstruction, at: instIndex)
        /// Remove all uses with the new instruction
        bb.parent.replaceAllUses(of: inst, with: %newInstruction)
        /// Remove all intermediate nodes
        removeIntermediates()
    }
}

open class AlgebraicExpressionAnalysis : AnalysisPass {
    public typealias Body = BasicBlock

    /// Extract an independent algebraic subexpression from instruction
    private static func subexpression(from inst: Instruction,
                                      visited: inout Set<Instruction>) throws -> AlgebraicExpression {
        /// Get user analysis
        let bb = inst.parent
        let function = bb.parent
        let users = try function.analysis(from: UserAnalysis.self)
        /// DFS expression builder
        func subexpression(from use: Use, isEntry: Bool = false) throws -> AlgebraicExpression {
            /// If it's not an instruction in the current basic block, it's an atom
            guard case let .instruction(_, inst) = use, inst.parent == bb else {
                return .atom(use)
            }
            /// Mark as visited
            visited.insert(inst)
            /// Treat nodes with users as atoms when and only when they are not the entry
            /// to this analysis
            if !isEntry && !users[inst].isEmpty {
                return .atom(%inst)
            }
            /// DFS from math instructions
            switch inst.kind {
            case let .map(op, v):
                return .unary(op, try subexpression(from: v), inst)
            case let .zipWith(op, lhs, rhs):
                return .binary(op, try subexpression(from: lhs),
                                   try subexpression(from: rhs), inst)
            case let .matrixMultiply(lhs, rhs):
                return .matrixMultiply(try subexpression(from: lhs),
                                       try subexpression(from: rhs), inst)
            case let .transpose(v):
                return .transpose(try subexpression(from: v), inst)
            default:
                return .atom(use)
            }
        }
        return try subexpression(from: %inst, isEntry: true)
    }

    /// Run pass on the basic block
    open static func run(on body: BasicBlock) throws -> [AlgebraicExpression] {
        var exprs: [AlgebraicExpression] = []
        var visited: Set<Instruction> = []
        /// Perform DFS for every unvisited instruction
        for inst in body.reversed() where !visited.contains(inst) {
            exprs.append(try subexpression(from: inst, visited: &visited))
        }
        return exprs
    }
}
