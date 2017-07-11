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

/// AlgebraicExpression is designed to simplify pattern matching for *independent*
/// math expressions in domain-specific optimizations such as Algebra Simplification,
/// Linear Algebra Fusion and Matrix Chain Ordering

public indirect enum AlgebraicExpression {
    case atom(Use)
    case map(UnaryOp, AlgebraicExpression, Instruction)
    case zipWith(BinaryOp, AlgebraicExpression, AlgebraicExpression, Instruction)
    case matrixMultiply(AlgebraicExpression, AlgebraicExpression, Instruction)
    case transpose(AlgebraicExpression, Instruction)
}

extension AlgebraicExpression {
    /// Top instruction (post-dominator) in the expression
    var topInstruction: Instruction? {
        switch self {
        case .atom(_):
            return nil
        case .zipWith(_, _, _, let inst),
             .matrixMultiply(_, _, let inst),
             .transpose(_, let inst),
             .map(_, _, let inst):
            return inst
        }
    }

    /// Remove intermediate instructions from the basic block
    func removeIntermediates(upTo barriers: AlgebraicExpression...) {
        let barriers = Set(barriers.lazy.flatMap { $0.topInstruction })
        func remove(_ expr: AlgebraicExpression) {
            if let inst = expr.topInstruction, barriers.contains(inst) {
                return
            }
            switch self {
            case .atom(_):
                return
            case let .zipWith(_, lhs, rhs, inst),
                 let .matrixMultiply(lhs, rhs, inst):
                remove(lhs)
                remove(rhs)
                inst.removeFromParent()
            case let .transpose(x, inst),
                 let .map(_, x, inst):
                remove(x)
                inst.removeFromParent()
            }
        }
        remove(self)
    }

    static prefix func % (expr: AlgebraicExpression) -> Use {
        switch expr {
        case .atom(let v): return v
        case .map(_, _, let inst),
             .matrixMultiply(_, _, let inst),
             .transpose(_, let inst),
             .zipWith(_, _, _, let inst):
            return %inst
        }
    }

    var value: Value {
        return (%self).value
    }

    static func ~= (pattern: IntegerLiteralType, expression: AlgebraicExpression) -> Bool {
        guard case let .atom(x) = expression else { return false }
        return pattern ~= x
    }

    static func ~= (pattern: FloatLiteralType, expression: AlgebraicExpression) -> Bool {
        guard case let .atom(x) = expression else { return false }
        return pattern ~= x
    }

    func makeLiteral(_ literal: Literal) -> LiteralValue {
        return value.makeLiteral(literal)
    }

    var isAtom: Bool {
        switch self {
        case .atom(_): return true
        default: return false
        }
    }
}

extension AlgebraicExpression : BackwardGraphNode {
    public var predecessors: [AlgebraicExpression] {
        switch self {
        case .atom(_):
            return []
        case .map(_, let x, _),
             .transpose(let x, _):
            return [x]
        case .matrixMultiply(let x, let y, _),
             .zipWith(_, let x, let y, _):
            return [x, y]
        }
    }
}

extension AlgebraicExpression : Equatable {
    public static func == (lhs: AlgebraicExpression, rhs: AlgebraicExpression) -> Bool {
        switch (lhs, rhs) {
        case _ where lhs.topInstruction == rhs.topInstruction:
            return true
        case let (.atom(x), .atom(y)) where x == y:
            return true
        default:
            return false
        }
    }
}

open class AlgebraicExpressionAnalysis : AnalysisPass {
    public typealias Body = BasicBlock

    private static func independentSubexpression(from inst: Instruction,
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
            /// Treat nodes with more than one users as atoms when and only when
            /// they are not the entry to this analysis
            if !isEntry && users[inst].count > 1 {
                return .atom(%inst)
            }
            /// DFS from math instructions
            switch inst.kind {
            case let .map(op, v):
                return .map(op, try subexpression(from: v), inst)
            case let .zipWith(op, lhs, rhs):
                return .zipWith(op, try subexpression(from: lhs),
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
            let subExpr = try independentSubexpression(from: inst, visited: &visited)
            exprs.append(subExpr)
        }
        return exprs
    }
}
