//
//  ValueLowering.swift
//  DLVMCodeGen
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

import CoreTensor
import DLVM
import LLVM_C

extension DLVM.Use : LLEmittable {
    public typealias LLUnit = LLVMValueRef
    @discardableResult
    public func emit<T>(to context: LLGenContext<T>,
                     in env: LLGenEnvironment) -> LLVMValueRef {
        switch self {
        case let .variable(_, val):
            return env.value(for: val)
        case let .function(_, fun):
            return env.value(for: fun)
        case let .argument(_, arg):
            return env.value(for: arg)
        case let .instruction(_, inst):
            return env.value(for: inst)
        case let .literal(ty, lit):
            return emitLiteral(lit, ofType: ty, to: context, in: env)
        }
    }

    @discardableResult
    private func emitLiteral<T>(_ literal: Literal, ofType type: Type,
                             to context: LLGenContext<T>,
                             in env: LLGenEnvironment) -> LLUnit {
        switch literal {
        case let .scalar(lit):
            switch lit {
            case let .bool(v):
                return v.constant
            case let .int(v):
                let llType = type.emit(to: context, in: env)
                return v.llValue(ofType: llType)
            case let .float(v):
                let llType = type.emit(to: context, in: env)
                return v.llValue(ofType: llType)
            }

        case let .array(xx):
            let vals = xx.map{$0.emit(to: context, in: env)}
            return vals.array(ofType: type.emit(to: context, in: env))

        case let .tuple(xx):
            return <%>xx.map{$0.emit(to: context, in: env)}

        case let .tensor(xx):
            let llType = type.emit(to: context, in: env)
            let elementTy = LLVMGetElementType(llType) ?? DLImpossibleResult()
            return xx.map { $0.emit(to: context, in: env) }.array(ofType: elementTy)

        case let .struct(fields):
            /// Sanity check
            guard case let .struct(structTy) = type else { DLImpossible() }
            DLAssert(structTy.fields.count == fields.count)
            for (tyField, useField) in zip(structTy.fields, fields) {
                let (formalName, formalType) = tyField
                let (actualName, actualUse) = useField
                DLAssert(formalName == actualName && formalType == actualUse.type)
            }
            /// Emit LLVM IR
            var elements: [LLVMValueRef?] = fields.map { $0.1.emit(to: context, in: env) }
            return LLVMConstStruct(&elements, UInt32(fields.count), .false)

        case .zero:
            /// - Note: There doesn't seem to be a way to emit LLVM's `zeroinitializer` with
            /// the C API, so we are doing a `bitcast`.
            let llType = type.emit(to: context, in: env)
            let size = LLVMSizeOf(llType)
            let zero = LLVMConstInt(size, 0, .false)
            return LLVMConstBitCast(zero, llType)

        case .undefined:
            return LLVMGetUndef(type.emit(to: context, in: env))

        case .null:
            return LLVMConstNull(type.emit(to: context, in: env))
        }
    }
}
