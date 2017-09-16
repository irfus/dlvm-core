//
//  BLAS.swift
//  DLVMCodeGen
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

/// Common BLAS
public enum BLAS {
    public struct SelfIncrementTerm {
        public let beta: Use
        public let value: Use
    }
    /// Level 1: c <- (+ (* a x) b)
    case axpy(a: Use, x: Use, b: Use, dataType: DataType)
    /// Level 2: C <- (+ (* alpha (• A x)) (* beta C))
    case gemv(alpha: Use, A: Use, trans: Bool, x: Use,
              increment: SelfIncrementTerm?, dataType: DataType)
    /// Level 3: C <- (+ (* alpha (• A B)) (* beta C))
    case gemm(alpha: Use, A: Use, trans: Bool, B: Use,
              increment: SelfIncrementTerm?, dataType: DataType)
}

public extension BLAS {

    static func subroutine(from entry: Instruction) throws -> (BLAS, Set<Instruction>)? {
        let bb = entry.parent
        /// Get all algebraic expressions in the current BB
        let algExpr = bb.analysis(from: AlgebraicExpressionAnalysis.self)
        /// Traverse along forward data flow from `entry`, find patterns that:
        /// - Matches one of the BLAS's
        /// - Contains `entry` as an instruction
        for inst in entry.traversed(in: .breadthFirst) {
            let expr = algExpr.expression(for: entry)
            switch expr {
            /// AXPY
            case .numericBinary(.add,
                                .numericBinary(.multiply,
                                               let a, let x, let times),
                          let b, let plus):
                switch (a.type, x.type, b.type) {
                case let (.tensor([], dt1), .tensor(s2, dt2), .tensor(s3, dt3))
                    where s2 == s3 && !s2.isScalar && dt1 == dt2 && dt2 == dt3:
                    let subroutine = BLAS.axpy(a: %a, x: %x, b: %b,
                                               dataType: dt1)
                    let insts: Set<Instruction> = [times, plus]
                    guard insts.contains(inst) else { break }
                    return (subroutine, insts)
                default:
                    break
                }
            /// GEMM with alpha
            case .numericBinary(.multiply, let alpha,
                                .dot(let A, let x, let dot), let times):
                switch (alpha.type, A.type, x.type) {
                case let (.tensor([], dt1), .tensor(s2, dt2), .tensor(s3, dt3))
                    where dt1 == dt2 && dt2 == dt3 && s2.isMatrix && s3.isMatrix:
                    var insts: Set<Instruction> = [dot, times]
                    /// See if A is transposed
                    let subroutine: BLAS
                    if case let .transpose(AT, trans) = A {
                        subroutine = BLAS.gemv(
                            alpha: %alpha, A: %AT, trans: true, x: %x,
                            increment: nil, dataType: dt1)
                        insts.insert(trans)
                    }
                    else {
                        subroutine = BLAS.gemv(
                            alpha: %alpha, A: %A, trans: false, x: %x,
                            increment: nil, dataType: dt1)
                    }
                    guard insts.contains(inst) else { break }
                    return (subroutine, insts)
                default:
                    break
                }
            /// GEMM without alpha
            case .dot(let A, let x, let dot):
                switch (A.type, x.type) {
                case let (.tensor(s2, dt2), .tensor(s3, dt3))
                    where dt2 == dt3 && s2.isMatrix && s3.isMatrix:
                    var insts: Set<Instruction> = [dot]
                    /// See if A is transposed
                    let subroutine: BLAS
                    if case let .transpose(AT, trans) = A {
                        subroutine = BLAS.gemv(
                            alpha: %x.makeScalar(1), A: %AT, trans: true, x: %x,
                            increment: nil, dataType: dt2)
                        insts.insert(trans)
                    }
                    else {
                        subroutine = BLAS.gemv(
                            alpha: %x.makeScalar(1), A: %A, trans: false, x: %x,
                            increment: nil, dataType: dt2)
                    }
                    guard insts.contains(inst) else { break }
                    return (subroutine, insts)
                default:
                    break
                }
            }
        }
        /// Not found
        return nil
    }
}
