//
//  NVVM.swift
//  DLVM
//
//  Created by Richard Wei on 3/21/17.
//
//

import LLVM_C

public struct NVVM : Target {
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

    public let module: LLVMModuleRef
    fileprivate var functions: [Intrinsic : LLVMValueRef] = [:]

    init(module: LLVMModuleRef) {
        self.module = module
    }
}

extension NVVM.Intrinsic : IRIntrinsic {
    public var type: LLVMTypeRef {
        switch self {
        case .barrier: // () -> void
            return LLVMFunctionType(LLVMVoidType(), nil, 0, 0)
        case _: // () -> i32
            return LLVMFunctionType(LLVMInt32Type(), nil, 0, 0)
        }
    }
}

extension NVVM {
    public subscript(intrinsic: Intrinsic) -> LLVMValueRef {
        mutating get {
            if let fun = functions[intrinsic] {
                return fun
            }
            let utf8 = intrinsic.rawValue.utf8Start
            let fun = utf8.withMemoryRebound(to: Int8.self, capacity: 1, { name in
                LLVMAddFunction(module, name, intrinsic.type)!
            })
            functions[intrinsic] = fun
            return fun
        }
    }
}
