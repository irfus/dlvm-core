//
//  Adjoints.swift
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

import struct CoreTensor.TensorShape

extension InstructionKind {
    public func operandAdjoints(
        using bd: IRBuilder, primal: Use, seed: Use
    ) -> [(operand: Use, adjoint: Use)]? {
        func unbroadcastIfNeeded(_ x: Use, to y: Use) -> Use {
            guard case let .tensor(s1, _) = x.type else {
                fatalError("\(x) is not a tensor")
            }
            guard case let .tensor(s2, _) = y.type else {
                fatalError("\(y) is not a tensor")
            }
            let reductionAxes = s1.indices
                .filter { $0 >= s2.count || s1[s1.count - $0 - 1] < s2[s2.count - $0 - 1] }
                .map { s1.count - $0 - 1 }
            /// If no need to sum, return original x
            if reductionAxes.isEmpty { return x }
            let sum = bd.reduce(.numeric(.add), x,
                                initial: %x.makeScalar(0), dims: reductionAxes)
            guard case let .tensor(s3, _) = sum.type else { DLImpossible() }
            /// If no need to shape cast, return sum
            if s2 == s3 { return %sum }
            /// Return shape-casted sum
            return %bd.shapeCast(%sum, to: s2)
        }

        var adjoints: [(operand: Use, adjoint: Use)]
        switch self {
        /** Literal constructor **/
        case let .literal(lit, _):
            adjoints = lit.operands.map {
                ($0, $0.makeLiteral(0, using: bd).makeUse())
            }

        /** Unary elementwise operations **/
        case let .numericUnary(.log, x):
            adjoints = [
                /// ∂f/∂x = D / x
                (x, %bd.divide(seed, x))
            ]
        case let .numericUnary(.cos, x):
            adjoints = [
                /// ∂f/∂x = -D * sin(x)
                (x, %bd.multiply(%bd.numeric(.negate, seed), %bd.numeric(.sin, x)))
            ]
        case let .numericUnary(.sin, x):
            adjoints = [
                /// ∂f/∂x = D * cos(x)
                (x, %bd.multiply(seed, %bd.numeric(.cos, x)))
            ]
        case let .numericUnary(.tan, x):
            let cosx = %bd.numeric(.cos, x)
            adjoints = [
                /// ∂f/∂x = D / (cos(x) * cos(x))
                (x, %bd.divide(seed, %bd.multiply(cosx, cosx)))
            ]
        case let .numericUnary(.cosh, x):
            adjoints = [
                /// ∂f/∂x = D * sinh(x)
                (x, %bd.multiply(seed, %bd.numeric(.sinh, x)))
            ]
        case let .numericUnary(.sinh, x):
            adjoints = [
                /// ∂f/∂x = D * cosh(x)
                (x, %bd.multiply(seed, %bd.numeric(.cosh, x)))
            ]
        case let .numericUnary(.tanh, x):
            adjoints = [
                /// ∂f/∂x = D * (1 - (f * f))
                (x, %bd.multiply(seed, %bd.subtract(%x.makeScalar(1), %bd.multiply(primal, primal))))
            ]
        case let .numericUnary(.acos, x):
            adjoints = [
                /// ∂f/∂x = -D / sqrt(1 - (x * x))
                (x, %bd.divide(
                    %bd.numeric(.negate, seed),
                    %bd.numeric(.sqrt, %bd.subtract(%x.makeScalar(1), %bd.multiply(x, x))))
                )
            ]
        case let .numericUnary(.asin, x):
            adjoints = [
                /// ∂f/∂x = D / sqrt(1 - (x * x))
                (x, %bd.divide(
                    seed,
                    %bd.numeric(.sqrt, %bd.subtract(%x.makeScalar(1), %bd.multiply(x, x))))
                )
            ]
        case let .numericUnary(.atan, x):
            adjoints = [
                /// ∂f/∂x = D / (1 + (x * x))
                (x, %bd.divide(seed, %bd.add(%x.makeScalar(1), %bd.multiply(x, x))))
            ]
        case let .numericUnary(.exp, x):
            adjoints = [
                /// ∂f/∂x = f * D
                (x, %bd.multiply(primal, seed))
            ]
        case let .numericUnary(.sqrt, x):
            adjoints = [
                /// ∂f/∂x = D / (2 * f)
                (x, %bd.divide(seed, %bd.multiply(%x.makeScalar(2), primal)))
            ]
        case .numericUnary:
            /// Adjoints for remaining cases not yet implemented
            DLUnimplemented()

        /** Binary elementwise operations **/
        case let .numericBinary(.add, lhs, rhs):
            adjoints = [
                /// ∂f/∂x = D
                (lhs, unbroadcastIfNeeded(seed, to: lhs)),
                /// ∂f/∂y = D
                (rhs, unbroadcastIfNeeded(seed, to: rhs))
            ]
        case let .numericBinary(.subtract, lhs, rhs):
            adjoints = [
                /// ∂f/∂x = D
                (lhs, unbroadcastIfNeeded(seed, to: lhs)),
                /// ∂f/∂y = -D
                (rhs, unbroadcastIfNeeded(%bd.numeric(.negate, seed), to: rhs))
            ]
        case let .numericBinary(.multiply, lhs, rhs):
            adjoints = [
                /// ∂f/∂x = D * y
                (lhs, unbroadcastIfNeeded(%bd.multiply(seed, rhs), to: lhs)),
                /// ∂f/∂y = D * x
                (rhs, unbroadcastIfNeeded(%bd.multiply(seed, lhs), to: rhs)),
            ]
        case let .numericBinary(.divide, lhs, rhs):
            adjoints = [
                /// ∂f/∂x = D / y
                (lhs, %bd.divide(seed, rhs)),
                /// ∂f/∂y = -x / y^2
                (rhs, %bd.numeric(.negate, %bd.divide(lhs, %bd.multiply(rhs, rhs))))
            ]
        case let .numericBinary(.power, lhs, rhs):
            adjoints = [
                /// ∂f/∂x = y * x^(y - 1) * D
                (lhs, %bd.multiply(
                    %bd.multiply(rhs, %bd.power(lhs, %bd.subtract(rhs, %rhs.makeScalar(1)))),
                    seed)
                ),
                /// ∂f/∂y = ln(x) * f * D
                (rhs, %bd.multiply(%bd.multiply(%bd.log(lhs), seed), seed))
            ]
        case .numericBinary(.min, _, _),
             .numericBinary(.max, _, _):
            /// Adjoint for min/max not yet implemented
            /// Implementation should include a special equality tiebreaker
            DLUnimplemented()

        case let .dataTypeCast(x, _):
            guard case let .tensor(_, xdt) = x.type else {
                fatalError("\(x) is not a tensor")
            }
            adjoints = [
                (x, %bd.dataTypeCast(seed, to: xdt))
            ]
        case let .dot(lhs, rhs):
            adjoints = [
                /// ∂f/∂x = D • y^T
                (lhs, %bd.dot(seed, %bd.transpose(rhs))),
                /// ∂f/∂y = x^T • D
                (rhs, %bd.dot(%bd.transpose(lhs), seed))
            ]
        case let .transpose(x):
            adjoints = [
                /// ∂f/∂x = D^T
                (x, %bd.transpose(seed))
            ]
        case let .reverse(x, dims):
            adjoints = [
                /// ∂f/∂x = reverse(D, dims)
                (x, %bd.reverse(seed, dims: dims))
            ]
        case let .slice(x, at: range):
            adjoints = [
                /// ∂f/∂x = slice(D, at: range)
                (x, %bd.slice(seed, at: range))
            ]

        case .compare:
            DLUnimplemented()
        case .select:
            DLUnimplemented()
        case .scan:
            DLUnimplemented()
        case .reduce:
            DLUnimplemented()
        case .concatenate:
            DLUnimplemented()
        case .reduceWindow:
            DLUnimplemented()
        case .convolve:
            DLUnimplemented()

        /** Cost-free casts **/
        case let .padShape(x, at: i):
            /// When dimension is known to be 1, calculate adjoint using
            /// squeezeShape
            if i == 1 {
                adjoints = [
                    /// ∂f/∂x = squeezeShape(D, at: i)
                    (x, %bd.squeezeShape(seed, at: i))
                ]
            }
            /// Otherwise, sum over the dimension
            else {
                adjoints = [
                    /// ∂f/∂x = sum(D, along: i)
                    (x, %bd.reduce(.numeric(.add), seed,
                                   initial: %x.makeScalar(0), dims: [i]))
                ]
            }
        case let .squeezeShape(x, at: i):
            adjoints = [
                /// ∂f/∂x = padShape(D, at: i)
                (x, %bd.padShape(seed, at: i))
            ]
        case let .shapeCast(x, s):
            adjoints = [
                /// ∂f/∂x = shapecast(D, at: i)
                (x, %bd.shapeCast(seed, to: s))
            ]
        case let .bitCast(x, t):
            adjoints = [
                /// ∂f/∂x = bitcast(D, at: t)
                (x, %bd.bitCast(seed, to: t))
            ]

        /** Aggregate operations **/
        case let .extract(from: x, at: i):
            adjoints = [
                /// ∂f/∂x = extract(from: D, at: i)
                (x, %bd.extract(from: seed, at: i))
            ]

        /** Function application **/
        case .apply:
            /// Adjoint of function application is a special case that
            /// should be handled by the user.
            return nil

        /// The following instructions are not differentiable because they:
        /// - are related to control-flow
        /// - are boolean operations
        /// - are integer math operations
        /// - are random operations
        /// - are related to the stack data structure
        /// - access memory
        case .branch, .branchEnum, .conditional, .return,
             .not, .booleanBinary,
             .numericBinary(.modulo, _, _),
             .numericBinary(.floorDivide, _, _),
             .numericBinary(.truncateDivide, _, _),
             .random,
             .createStack, .destroyStack, .push, .pop,
             .insert, .allocateStack, .allocateHeap, .allocateBox, .projectBox,
             .retain, .release, .deallocate, .load, .store, .elementPointer, .copy,
             .trap:
            return nil
        }
        return adjoints
    }
}
