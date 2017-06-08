//
//  NVVM.swift
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

import LLVM_C
import DLVM

/// NVVM Target
public final class NVVM : ComputeTarget, LLFunctionPrototypeCacheable {

    public enum Intrinsic : StaticString {
        case threadIndexX    = "llvm.nvvm.read.ptx.sreg.tid.x"    // () -> i32
        case threadIndexY    = "llvm.nvvm.read.ptx.sreg.tid.y"    // () -> i32
        case threadIndexZ    = "llvm.nvvm.read.ptx.sreg.tid.z"    // () -> i32
        case blockIndexX     = "llvm.nvvm.read.ptx.sreg.ctaid.x"  // () -> i32
        case blockIndexY     = "llvm.nvvm.read.ptx.sreg.ctaid.y"  // () -> i32
        case blockIndexZ     = "llvm.nvvm.read.ptx.sreg.ctaid.z"  // () -> i32
        case blockDimensionX = "llvm.nvvm.read.ptx.sreg.ntid.x"   // () -> i32
        case blockDimensionY = "llvm.nvvm.read.ptx.sreg.ntid.y"   // () -> i32
        case blockDimensionZ = "llvm.nvvm.read.ptx.sreg.ntid.z"   // () -> i32
        case gridDimensionX  = "llvm.nvvm.read.ptx.sreg.nctaid.x" // () -> i32
        case gridDimensionY  = "llvm.nvvm.read.ptx.sreg.nctaid.y" // () -> i32
        case gridDimensionZ  = "llvm.nvvm.read.ptx.sreg.nctaid.z" // () -> i32
        case barrier         = "llvm.nvvm.barrier0"               // () -> void
    }

    public enum MemoryCopyKind : Int, LLConstantConvertible {
        case hostToHost = 0
        case hostToDevice = 1
        case deviceToHost = 2
        case deviceToDevice = 3

        public var llType: LLVMTypeRef {
            return i32
        }
    }

    public enum RuntimeFunction {
        case malloc(LLVMValueRef) // (i32) -> i8*
        case free(LLVMValueRef) // (i8*) -> void
        case synchronize // () -> i32
        case memcpy(to: LLVMValueRef, from: LLVMValueRef, count: LLVMValueRef, kind: MemoryCopyKind) // (i8*, i8*, i32, i32) -> i32
        case launchKernel(LLVMValueRef, grid: LLVMValueRef, block: LLVMValueRef, arguments: LLVMValueRef, sharedMemory: LLVMValueRef, stream: LLVMValueRef)
    }

    public enum RuntimeType {
        case stream
        case dimension
        case result
    }

    public let module: LLVMModuleRef
    private let builder: LLVMBuilderRef = LLVMCreateBuilder()
    public var functions: [AnyHashable : LLVMValueRef] = [:]

    public init(module: LLVMModuleRef) {
        self.module = module
    }

    deinit {
        LLVMDisposeBuilder(builder)
        LLVMDisposeModule(module)
    }
}

extension NVVM.Intrinsic : LLFunctionPrototype {
    public var type: LLVMValueRef {
        switch self {
        case .barrier: return [] => void
        case _: return [] => i32
        }
    }

    public var arguments: [LLVMValueRef] {
        return []
    }
}

extension NVVM.RuntimeType : LLTypePrototype {
    public var name: StaticString {
        switch self {
        case .dimension: return "dim3"
        case .stream: return "cudaStream_t"
        case .result: return "cudaError_t"
        }
    }

    public var type: LLVMTypeRef {
        switch self {
        case .dimension: return ^[i32, i32, i32]
        case .stream: return ^name.description
        case .result: return i32
        }
    }
}

extension NVVM.RuntimeFunction : LLFunctionPrototype {
    public var name: StaticString {
        switch self {
        case .malloc: return "cudaMalloc"
        case .free: return "cudaFree"
        case .memcpy: return "cudaMemcpy"
        case .synchronize: return "cudaDeviceSynchronize"
        case .launchKernel: return "cudaLaunchKernel"
        }
    }
    
    public var type: LLVMTypeRef {
        switch self {
        case .malloc: return [i32] => void
        case .free: return [i8*] => void
        case .memcpy: return [i8*, i8*, i32, i32] => i8*
        case .synchronize: return [] => void
        case .launchKernel:
            let dim3 = NVVM.RuntimeType.dimension.type
            let cudaStream_t = NVVM.RuntimeType.stream.type
            return [i8*, dim3, dim3, i8**, i32, cudaStream_t] => i32
        }
    }

    public var arguments: [LLVMValueRef] {
        switch self {
        case .synchronize: return []
        case let .malloc(v1): return [v1]
        case let .free(v1): return [v1]
        case let .memcpy(v1, v2, v3, kind):
            return [v1, v2, v3, kind.constant]
        case let .launchKernel(v1, v2, v3, v4, v5, v6):
            return [v1, v2, v3, v4, v5, v6]
        }
    }
}
