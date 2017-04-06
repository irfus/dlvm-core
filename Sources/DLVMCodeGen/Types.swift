//
//  Types.swift
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
import LLVM

public extension DataType {
    var llType: IRType {
        switch self {
        case .bool: return i1
        case let .int(w): return IntType(width: Int(w))
        case .float(.half): return FloatType.half
        case .float(.single): return FloatType.float
        case .float(.double): return FloatType.double
        }
    }
}

// MARK: - Boolean lowering
extension Bool {
    var constant: Constant<Signed> {
        return IntType.int1.constant(self ? 1 : 0)
    }
}

// MARK: - Constant convertible
/// - Note: This is used to create target-specific enum constants
public protocol LLConstantConvertible : RawRepresentable {
    associatedtype RawValue : SignedInteger
    var constantType: IntType { get }
}

public extension LLConstantConvertible {
    var constant: IRValue {
        return constantType.constant(rawValue)
    }
}
