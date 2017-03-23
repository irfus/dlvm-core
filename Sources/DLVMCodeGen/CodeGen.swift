//
//  CodeGen.swift
//  DLVM
//
//  Created by Richard Wei on 3/13/17.
//
//

import Foundation
import DLVM
import LLVM

public class CodeGenerator<TargetType : Target & FunctionPrototypeCache> {
    let context: LLVM.Context
    let builder: LLVM.IRBuilder
    let module: LLVM.Module

    let dlModule: DLVM.Module
    lazy var target: TargetType = TargetType(module: self.module)

    init(module: DLVM.Module) {
        self.dlModule = module
        self.context = LLVM.Context.global
        self.module = LLVM.Module(name: module.name)
        self.builder = LLVM.IRBuilder(module: self.module)
    }
}

extension CodeGenerator {
    func emit() {

    }
}

public extension CodeGenerator {
    func writeBitcode(toFile file: String) throws {
        try module.emitBitCode(to: file)
    }
}
