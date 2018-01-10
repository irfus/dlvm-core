//
//  AlgebraicExpression.swift
//  DLVM
//
//  Copyright 2016-2018 The DLVM Team.
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

/// AlgebraicExpression is designed to simplify pattern matching for
/// *independent* math expressions in domain-specific optimizations such as
/// Algebra Simplification, Linear Algebra Fusion and Matrix Multiplication
/// Reordering.

import CoreOp

public indirect enum AlgebraicExpression {
    case atom(Use)
    case map(NumericUnaryOp, AlgebraicExpression, Instruction)
    case numericBinary(NumericBinaryOp,
        AlgebraicExpression, AlgebraicExpression, Instruction)
    case booleanBinary(BooleanBinaryOp,
        AlgebraicExpression, AlgebraicExpression, Instruction)
    case dot(AlgebraicExpression, AlgebraicExpression, Instruction)
    case transpose(AlgebraicExpression, Instruction)
}

public struct AlgebraicRepresentation {
    public fileprivate(set) var expressions: [AlgebraicExpression] = []
    fileprivate var table: [Instruction : AlgebraicExpression] = [:]
}

public extension AlgebraicRepresentation {
    func expression(for instruction: Instruction) -> AlgebraicExpression {
        guard let expr = table[instruction] else {
            preconditionFailure("""
                Instruction \(instruction) does not belong to the basic block \
                where the analysis is performed on
                """)
        }
        return expr
    }

    func contains(_ instruction: Instruction) -> Bool {
        return table.keys.contains(instruction)
    }
}

public extension AlgebraicExpression {
    /// Top instruction (post-dominator) in the expression
    var topInstruction: Instruction? {
        switch self {
        case .atom(_):
            return nil
        case .numericBinary(_, _, _, let inst),
             .booleanBinary(_, _, _, let inst),
             .dot(_, _, let inst),
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
            switch expr {
            case .atom(_):
                return
            case let .numericBinary(_, lhs, rhs, inst),
                 let .booleanBinary(_, lhs, rhs, inst),
                 let .dot(lhs, rhs, inst):
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
             .dot(_, _, let inst),
             .transpose(_, let inst),
             .numericBinary(_, _, _, let inst),
             .booleanBinary(_, _, _, let inst):
            return %inst
        }
    }

    var value: Value {
        return (%self).value
    }

    var type: Type {
        return (%self).type
    }

    static func ~= (pattern: IntegerLiteralType,
                    expression: AlgebraicExpression) -> Bool {
        guard case let .atom(x) = expression else { return false }
        return pattern ~= x
    }

    static func ~= (pattern: FloatLiteralType,
                    expression: AlgebraicExpression) -> Bool {
        guard case let .atom(x) = expression else { return false }
        return pattern ~= x
    }

    func makeLiteral(_ literal: Literal,
                     before inst: Instruction,
                     using builder: IRBuilder) -> Value {
        return value.makeLiteral(literal, before: inst, using: builder)
    }

    func makeScalar(_ scalar: Literal.Scalar) -> LiteralValue {
        return value.makeScalar(scalar)
    }

    var isAtom: Bool {
        switch self {
        case .atom(_): return true
        default: return false
        }
    }

    var instructions: AnySequence<Instruction> {
        return AnySequence(
            transposeTraversed(in: .breadthFirst).lazy.flatMap { expr in
                expr.topInstruction
            })
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
        case .dot(let x, let y, _),
             .numericBinary(_, let x, let y, _),
             .booleanBinary(_, let x, let y, _):
            return [x, y]
        }
    }
}

extension AlgebraicExpression : Equatable {
    public static func == (lhs: AlgebraicExpression,
                           rhs: AlgebraicExpression) -> Bool {
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

extension AlgebraicExpression : CustomStringConvertible {
    public var description: String {
        switch self {
        case let .atom(x):
            return "[\(x)]"
        case let .map(op, exp, _):
            return "(\(op) \(exp))"
        case let .dot(lhs, rhs, _):
            return "(dot \(lhs) \(rhs))"
        case let .transpose(exp, _):
            return "(transpose \(exp))"
        case let .numericBinary(op, lhs, rhs, _):
            return "(\(op) \(lhs) \(rhs))"
        case let .booleanBinary(op, lhs, rhs, _):
            return "(\(op) \(lhs) \(rhs))"
        }
    }
}

open class AlgebraicExpressionAnalysis : AnalysisPass {
    public typealias Body = BasicBlock
    public typealias Result = AlgebraicRepresentation

    private static func collect(
        from inst: Instruction,
        to repr: inout AlgebraicRepresentation) {
        /// Get user analysis
        let bb = inst.parent
        let function = bb.parent
        let dfg = function.analysis(from: DataFlowGraphAnalysis.self)
        /// DFS expression builder
        func subexpression(
            from use: Use,
            isEntry: Bool = false) -> AlgebraicExpression {
            /// If not an instruction in the current basic block, it's an atom
            guard case let .instruction(_, inst) = use, inst.parent == bb else {
                return .atom(use)
            }
            /// Treat nodes with more than one users as atoms when and only when
            /// they are not the ento this analysis
            if !isEntry && dfg.successors(of: inst).count > 1 {
                return .atom(%inst)
            }
            /// DFS from math instructions
            let expr: AlgebraicExpression
            switch inst.kind {
            case let .numericUnary(op, v):
                expr = .map(op, subexpression(from: v), inst)
            case let .numericBinary(op, lhs, rhs):
                expr = .numericBinary(op, subexpression(from: lhs),
                                subexpression(from: rhs), inst)
            case let .booleanBinary(op, lhs, rhs):
                expr = .booleanBinary(op, subexpression(from: lhs),
                                      subexpression(from: rhs), inst)
            case let .dot(lhs, rhs):
                expr = .dot(subexpression(from: lhs),
                            subexpression(from: rhs), inst)
            case let .transpose(v):
                expr = .transpose(subexpression(from: v), inst)
            default:
                expr = .atom(use)
            }
            repr.table[inst] = expr
            return expr
        }
        repr.expressions.append(subexpression(from: %inst, isEntry: true))
    }

    /// Run pass on the basic block
    open static func run(on body: BasicBlock) -> AlgebraicRepresentation {
        var repr = AlgebraicRepresentation()
        /// Perform DFS for every unvisited instruction
        for inst in body.reversed() where !repr.contains(inst) {
            collect(from: inst, to: &repr)
        }
        return repr
    }
}
