//
//  CodeGen.swift
//  DLVM
//
//  Created by Richard Wei on 3/13/17.
//
//

import Foundation
import DLVM
import LLVM_C

public class CodeGenerator {
    let llContext: LLVMContextRef
    let llBuilder: LLVMBuilderRef
    let llModule: LLVMModuleRef

    let module: DLVM.Module

    init(module: DLVM.Module) {
        self.module = module
        self.llContext = LLVMGetGlobalContext()
        self.llModule = LLVMModuleCreateWithName(module.name)
        self.llBuilder = LLVMCreateBuilderInContext(llContext)
    }
}

extension CodeGenerator {
    func emit() {
        
    }
}

public extension CodeGenerator {
    func writeBitcode(to file: FileHandle) {
        LLVMWriteBitcodeToFD(llModule, file.fileDescriptor, 0, 0)
    }
}
