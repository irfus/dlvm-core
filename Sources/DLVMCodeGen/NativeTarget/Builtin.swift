//
//  Builtin.swift
//  DLVM
//
//  Copyright 2016-2017 The DLVM Team.
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

import LLVM_C
import DLVM

public class Builtin : LLFunctionPrototypeCacheable {
    public var functions: [AnyHashable : LLVMValueRef] = [:]
    public var module: LLVMModuleRef
    public required init(module: LLVMModuleRef) {
        self.module = module
    }
}

extension Builtin : NativeTarget {
    public static var pointerSize: Int {
        return 8
    }
}

/// Runtime reference counter type
var referenceCounterType: LLVMTypeRef {
    DLUnimplemented()
}

/// Stack type
var stackType: LLVMTypeRef {
    DLUnimplemented()
}

public extension Builtin {
    enum Memory {
        case memcpy(to: LLVMValueRef, from: LLVMValueRef, count: LLVMValueRef, align: LLVMValueRef, isVolatile: LLVMValueRef)
        case malloc(size: LLVMValueRef)
        case free(LLVMValueRef)
        case lifetimeStart(size: LLVMValueRef, pointer: LLVMValueRef)
        case lifetimeEnd(size: LLVMValueRef, pointer: LLVMValueRef)
    }

    enum AccessOwner : Int32, LLConstantConvertible {
        case host = 1
        case device = 2
        
        public var constantType: LLVMTypeRef { return i32 }
    }

    enum Reference {
        case initialize(LLVMValueRef, AccessOwner)
        case retain(LLVMValueRef)
        case release(LLVMValueRef)
        case deallocate(LLVMValueRef)
    }

    enum RuntimeType : StaticString, LLTypePrototype {
        case memoryTracker = "class.DLMemoryTracker"
        case reference = "struct.DLReference"
        case runtimeRoutines = "struct.DLDeviceRuntimeRoutines"
    }

    enum MemoryTracker : LLFunctionPrototype {
        case create(LLVMValueRef)
        case destroy(LLVMValueRef)
        case requestMemory(LLVMValueRef, LLVMValueRef, AccessOwner)
        case registerMemory(LLVMValueRef, LLVMValueRef)
        case unregisterMemory(LLVMValueRef, LLVMValueRef)
        case setOutOfSync(LLVMValueRef, LLVMValueRef)
        case switchToHost(LLVMValueRef, LLVMValueRef)
        case clear(LLVMValueRef)

        public var name: StaticString {
            switch self {
            case .clear: return "DLMemoryTrackerClear"
            case .create: return "DLMemoryTrackerCreate"
            case .destroy: return "DLMemoryTrackerDestroy"
            case .registerMemory: return "DLMemoryTrackerRegisterMemory"
            case .setOutOfSync: return "DLMemoryTrackerSetOutOfSync"
            case .switchToHost: return "DLMemoryTrackerSwitchToHost"
            case .requestMemory: return "DLMemoryTrackerRequestMemory"
            case .unregisterMemory: return "DLMemoryTrackerRegisterMemory"
            }
        }

        public var type: LLVMTypeRef {
            switch self {
            case .clear:
                return [RuntimeType.memoryTracker.type*] => void
            case .create:
                return [] => RuntimeType.memoryTracker.type
            case .destroy:
                return [RuntimeType.memoryTracker.type*] => void
            case .registerMemory:
                return [RuntimeType.memoryTracker.type*, i8*, i64] => void
            case .setOutOfSync:
                return [RuntimeType.memoryTracker.type*, i8*] => void
            case .switchToHost:
                return [RuntimeType.memoryTracker.type*, i8*] => void
            case .requestMemory:
                return [RuntimeType.memoryTracker.type*, i8*, i32] => void
            case .unregisterMemory:
                return [RuntimeType.memoryTracker.type*, i8*] => void
            }
        }

        public var arguments: [LLVMValueRef] {
            switch self {
            case let .clear(obj):
                return [obj]
            case let .create(routines):
                return [routines]
            case let .destroy(obj):
                return [obj]
            case let .registerMemory(obj, ptr):
                return [obj, ptr]
            case let .setOutOfSync(obj, ptr):
                return [obj, ptr]
            case let .switchToHost(obj, ptr):
                return [obj, ptr]
            case let .requestMemory(obj, ptr, owner):
                return [obj, ptr, %owner]
            case let .unregisterMemory(obj, ptr):
                return [obj, ptr]
            }
        }
    }

    enum Control {
        case trap
    }
}

extension Builtin.Memory : LLFunctionPrototype {
    public var name: StaticString {
        switch self {
        case .memcpy: return "llvm.memcpy.p0i8.p0i8.i64"
        case .malloc: return "malloc"
        case .free: return "free"
        case .lifetimeStart: return "llvm.lifetime.start"
        case .lifetimeEnd: return "llvm.lifetime.end"
        }
    }

    public var arguments: [LLVMValueRef] {
        switch self {
        case let .free(val):
            return [val]
        case let .malloc(size: val):
            return [val]
        case let .memcpy(to: v1, from: v2, count: v3, align: v4, isVolatile: v5):
            return [v1, v2, v3, v4, v5]
        case let .lifetimeStart(size: size, pointer: ptr):
            return [size, ptr]
        case let .lifetimeEnd(size: size, pointer: ptr):
            return [size, ptr]
        }
    }
    
    public var type: LLVMTypeRef {
        switch self {
        case .free: return [i8*] => void
        case .malloc: return [i64*] => i8*
        case .memcpy: return [i8*, i8*, i64, i32, i1] => void
        case .lifetimeStart: return [i64, i8*] => void
        case .lifetimeEnd: return [i64, i8*] => void
        }
    }
}

extension Builtin.Reference : LLFunctionPrototype {

    public var name: StaticString {
        switch self {
        case .initialize: return "DLReferenceInit"
        case .retain: return "DLReferenceRetain"
        case .release: return "DLReferenceRelease"
        case .deallocate: return "DLReferenceDeallocate"
        }
    }

    public var arguments: [LLVMValueRef] {
        switch self {
        case let .initialize(v1, v2):
            return [v1, %v2]
        case let .retain(v):
            return [v]
        case let .release(v):
            return [v]
        case let .deallocate(v):
            return [v]
        }
    }

    public var type: LLVMTypeRef {
        switch self {
        case .initialize:
            return [^[], i32] => referenceCounterType
        case .retain, .release, .deallocate:
            return [referenceCounterType*] => void
        }
    }
}

extension Builtin.Control : LLFunctionPrototype {
    public var name: StaticString {
        switch self {
        case .trap: return "llvm.trap"
        }
    }

    public var arguments: [LLVMValueRef] {
        switch self {
        case .trap: return []
        }
    }

    public var type: LLVMTypeRef {
        switch self {
        case .trap: return [] => void
        }
    }
}
