//
//  Environment.swift
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
import LLVM_C

protocol LLEmittable {
    associatedtype LLUnit
    @discardableResult func emit<T>(to context: inout LLGenContext<T>,
                                    in env: inout LLGenEnvironment) -> LLUnit
}

/// Environment contains mappings from DLVM definitions to LLVM definitions
struct LLGenEnvironment {
    fileprivate var globals: [AnyHashable : LLVMValueRef] = [:]
    fileprivate var locals: [AnyHashable : LLVMValueRef] = [:]
    fileprivate var types: [AnyHashable : LLVMTypeRef] = [:]
    fileprivate var basicBlocks: [AnyHashable : LLVMBasicBlockRef] = [:]
}

extension LLGenEnvironment {
    mutating func clearLocals() {
        locals.removeAll()
    }

    mutating func insertGlobal<T : Definition & Hashable>
        (_ value: LLVMValueRef, for dlValue: T) {
        globals[dlValue] = value
    }

    mutating func insertLocal<T : Definition & Hashable>
        (_ value: LLVMValueRef, for dlValue: T) {
        locals[dlValue] = value
    }

    mutating func insertType(_ loweredType: LLVMTypeRef,
                             for type: AnyHashable) {
        types[type] = loweredType
    }

    mutating func insertBasicBlock(_ basicBlock: LLVMBasicBlockRef,
                                   for dlvmBasicBlock: BasicBlock) {
        basicBlocks[dlvmBasicBlock] = basicBlock
    }

    func value<T : Definition & Hashable>(for value: T) -> LLVMValueRef {
        return locals[value] ?? globals[value] ?? DLImpossibleResult()
    }

    func type(for alias: AnyHashable) -> LLVMTypeRef {
        return types[alias] ?? DLImpossibleResult()
    }

    func basicBlock(for dlvmBasicBlock: BasicBlock) -> LLVMBasicBlockRef {
        return basicBlocks[dlvmBasicBlock] ?? DLImpossibleResult()
    }
}

/// Context contains module, target, builder, etc
public class LLGenContext<TargetType : ComputeTarget> {
    public let dlModule: DLVM.Module
    public let module: LLVMModuleRef
    public let target: TargetType
    public let builder: LLVMBuilderRef
    lazy var builtin: Builtin = Builtin(module: self.module)

    public init(module: DLVM.Module) {
        dlModule = module
        let context = LLVMGetGlobalContext()
        self.module = dlModule.name.withCString { ptr in
            LLVMModuleCreateWithNameInContext(ptr, context)
        }
        target = TargetType(module: self.module)
        builder = LLVMCreateBuilderInContext(context)
    }

    deinit {
        LLVMDisposeModule(module)
        LLVMDisposeBuilder(builder)
    }
}
