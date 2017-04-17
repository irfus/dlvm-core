//
//  Builtin.swift
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

import LLVM
import DLVM

public class Builtin {
    public var functions: [AnyHashable : LLVM.Function] = [:]
    public var module: LLVM.Module
    public required init(module: LLVM.Module) {
        self.module = module
    }
}

/// Runtime reference counter type
var referenceCounterType: LLVM.StructType {
    return LLVM.StructType(elementTypes: [
        StructType.empty*,
        i64, // reference count
        i32, // memory type
    ])
}

public extension Builtin {
    enum Memory {
        case memcpy(to: IRValue, from: IRValue, count: IRValue, align: IRValue, isVolatile: IRValue)
        case malloc(size: IRValue)
        case free(IRValue)
    }

    enum AccessOwner : Int32, LLConstantConvertible {
        case none = 0
        case host = 1
        case device = 2
        
        public var constantType: IntType { return i32 }
    }

    enum Reference {
        case initialize(IRValue, AccessOwner)
        case retain(IRValue)
        case release(IRValue)
        case deallocate(IRValue)
        case accessOwner(IRValue)
        case setAccessOwner(IRValue, AccessOwner)
    }
}

extension Builtin.Memory : LLFunctionPrototype {
    public var name: StaticString {
        switch self {
        case .memcpy: return "llvm.memcpy.p0i8.p0i8.i64"
        case .malloc: return "malloc"
        case .free: return "free"
        }
    }

    public var arguments: [IRValue] {
        switch self {
        case let .free(val):
            return [val]
        case let .malloc(size: val):
            return [val]
        case let .memcpy(to: v1, from: v2, count: v3, align: v4, isVolatile: v5):
            return [v1, v2, v3, v4, v5]
        }
    }
    
    public var type: FunctionType {
        switch self {
        case .free: return [i8*] => void
        case .malloc: return [i64*] => i8*
        case .memcpy: return [i8*, i8*, i64, i32, i1] => void
        }
    }
}

extension Builtin.Reference : LLFunctionPrototype {

    public var name: StaticString {
        switch self {
        case .initialize: return "DLReferenceInit"
        case .retain: return "DLReferenceRetain"
        case .release: return "DLReferenceRelease"
        case .deallocate: return "DLVMReferenceDeallocate"
        case .accessOwner: return "DLReferenceGetAccessOwner"
        case .setAccessOwner: return "DLReferenceSetAccessOwner"
        }
    }

    public var arguments: [IRValue] {
        switch self {
        case let .initialize(v1, v2):
            return [v1, v2.constant]
        case let .retain(v):
            return [v]
        case let .release(v):
            return [v]
        case let .deallocate(v):
            return [v]
        case let .accessOwner(v):
            return [v]
        case let .setAccessOwner(v1, v2):
            return [v1, v2.constant]
        }
    }

    public var type: FunctionType {
        switch self {
        case .initialize:
            return [StructType.empty, i32] => referenceCounterType
        case .retain, .release, .deallocate:
            return [referenceCounterType*] => void
        case .accessOwner:
            return [referenceCounterType*] => i32
        case .setAccessOwner:
            return [referenceCounterType*, i32] => void
        }
    }
}
