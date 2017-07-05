//
//  Subroutine.swift
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

public enum BLAS {
    public struct SelfIncrementTerm {
        public let beta: Use
        public let value: Use
    }
    /// c <- a * x + b
    case axpy(a: Use, x: Use, b: Use)
    /// C <- alpha * Ax + beta * C
    case gemv(alpha: Use, A: Use, trans: Bool, x: Use, increment: SelfIncrementTerm?)
    /// C <- alpha * AB + beta * C   
    case gemm(alpha: Use, A: Use, trans: Bool, B: Use, increment: SelfIncrementTerm?)
}

protocol BLASCapable {
    static func blasFusion(from instruction: Instruction) -> (BLAS, [Instruction])?
}
