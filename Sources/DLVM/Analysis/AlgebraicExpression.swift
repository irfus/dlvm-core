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

open class AlgebraicExpressionAnalysis : AnalysisPass {
    public typealias Body = BasicBlock

    private static func subexpression(from use: Use,
                                      ignoringUsers: Bool = true) throws -> AlgebraicExpression {
        guard case let .instruction(_, inst) = use else {
            return .atom(use)
        }
        let function = inst.parent.parent
        let users = try function.analysis(from: UserAnalysis.self)
        if !ignoringUsers && !users[inst].isEmpty {
            return .atom(%inst)
        }
        switch inst.kind {
        case let .map(op, v):
            return .unary(op, try subexpression(from: v), inst)
        case let .zipWith(op, lhs, rhs):
            return try .binary(op, subexpression(from: lhs), subexpression(from: rhs), inst)
        case let .matrixMultiply(lhs, rhs):
            return try .matrixMultiply(subexpression(from: lhs), subexpression(from: rhs), inst)
        case let .transpose(v):
            return try .transpose(subexpression(from: v), inst)
        default:
            return .atom(use)
        }
    }

    open static func run(on body: BasicBlock) throws -> [AlgebraicExpression] {
        DLUnimplemented()
    }
}
