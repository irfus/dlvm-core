//
//  Statement.swift
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

import CoreOp

public indirect enum Statement {
    case `let`(String, Expression, body: Statement)
    case assert(condition: Expression, message: Expression)
    case produce(String, Statement)
    case store(Expression, index: Expression, predicate: Expression)
    case provide(Computation, index: Int, value: Expression, arguments: [Expression])
    case allocate(Variable, Type, extents: [Expression], condition: Expression)
    case free(Variable)
    case realize(
        Computation, index: Int, Type, bounds: CountableRange<UInt>,
        condition: Expression, body: Statement)
    case block([Statement])
    case `if`(Expression, then: Statement, `else`: Statement)
    case evaluate(Expression)
    case apply(Computation, [Expression])
    case call(CallKind, String, [Expression])
    case `for`(ForKind, Variable, range: CountableRange<UInt>, deviceAPI: DeviceAPI, body: Statement)
    case shuffle([Expression], indices: [Expression])
    case prefetch(Computation, index: Int, type: Type, bounds: CountableRange<UInt>)
}

public enum CallKind {
    case extern
    case externCXX
    case pureExtern
    case intrinsic
    case pureIntrinsic
}

public enum ForKind {
    case serial
    case parallel
    case vectorized
    case unrolled
}

public enum DeviceAPI {
    case cpu
    case cuda
    case openCL
    case metal
}
