//
//  NVVM.swift
//  DLVM
//
//  Created by Richard Wei on 3/21/17.
//
//

import LLVM

public final class NVVM : Target, FunctionPrototypeCache {
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

    public enum MemoryCopyKind : Int, IRValueConvertible {
        case hostToHost = 0
        case hostToDevice = 1
        case deviceToHost = 2
        case deviceToDevice = 3

        public var constantType: IntType {
            return i32
        }
    }

    public enum RuntimeFunction {
        case malloc(IRValue) // (i32) -> void*
        case free(IRValue) // (void*) -> void
        case synchronize // () -> i32
        case memcpy(to: IRValue, from: IRValue, count: IRValue, kind: MemoryCopyKind) // (void*, void*, i32, i32) -> i32
        case launchKernel(IRValue, grid: IRValue, block: IRValue, arguments: IRValue, sharedMemory: IRValue, stream: IRValue)
    }

    public enum RuntimeType {
        case stream
        case dimension
        case result
    }

    public unowned let module: LLVM.Module
    fileprivate lazy var builder: IRBuilder = IRBuilder(module: self.module)
    public var functions: [AnyHashable : Function] = [:]

    public init(module: LLVM.Module) {
        self.module = module
    }
}

extension NVVM.Intrinsic : FunctionPrototype {
    public var type: FunctionType {
        switch self {
        case .barrier: return [] => void
        case _: return [] => i32
        }
    }

    public var arguments: [IRValue] {
        return []
    }
}

extension NVVM.RuntimeType : TypePrototype {
    public var name: StaticString {
        switch self {
        case .dimension: return "dim3"
        case .stream: return "cudaStream_t"
        case .result: return "cudaError_t"
        }
    }

    public var type: IRType {
        switch self {
        case .dimension: return StructType(elementTypes: [i32, i32, i32])
        case .stream: return StructType(name: name.description)
        case .result: return i32
        }
    }
}

extension NVVM.RuntimeFunction : FunctionPrototype {
    public var name: StaticString {
        switch self {
        case .malloc: return "cudaMalloc"
        case .free: return "cudaFree"
        case .memcpy: return "cudaMemcpy"
        case .synchronize: return "cudaDeviceSynchronize"
        case .launchKernel: return "cudaLaunchKernel"
        }
    }
    
    public var type: FunctionType {
        switch self {
        case .malloc: return [i32] => void
        case .free: return [void*] => void
        case .memcpy: return [void*, void*, i32, i32] => void*
        case .synchronize: return [] => void
        case .launchKernel:
            let dim3 = NVVM.RuntimeType.dimension.type
            let cudaStream_t = NVVM.RuntimeType.stream.type
            return [void*, dim3, dim3, void**, i32, cudaStream_t] => i32
        }
    }

    public var arguments: [IRValue] {
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
