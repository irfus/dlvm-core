//
//  CodeGen.swift
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

/// Code generator protocol
/// - Note: This is often used as a target-erased existential for LLGen
public protocol CodeGenerator {
    func emitIR()
    func writeBitcode(toFile file: String) throws
    var textualIR: String { get }
}

/// LLGen, the unified LLVM code generator for DLVM modules
public class LLGen<TargetType :ComputeTarget> {
    public let dlModule: DLVM.Module
    public lazy internal(set) var context: LLGenContext<TargetType> = LLGenContext(module: self.dlModule)
    var environment: LLGenEnvironment = LLGenEnvironment()

    public init(module: DLVM.Module) {
        self.dlModule = module
        do { try module.verify() }
        catch { DLImpossible() }
    }
}

/// Error thrown during LLGen
public enum LLGenError : Error {
    case fileError(path: String)
}

extension LLGen : CodeGenerator {
    public func emitIR() {
        dlModule.emit(to: &context, in: &environment)
    }
    
    public func writeBitcode(toFile path: String) throws {
        guard LLVMWriteBitcodeToFile(context.module, path) == 0 else {
            throw LLGenError.fileError(path: path)
        }
    }

    public var textualIR: String {
        let cStr = LLVMPrintModuleToString(context.module)!
        return String(cString: cStr)
    }
}


extension DLVM.Use : LLEmittable {
    typealias LLUnit = LLVMValueRef
    @discardableResult
    func emit<T>(to context: inout LLGenContext<T>,
              in env: inout LLGenEnvironment) -> LLVMValueRef {
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
            return emitLiteral(lit, ofType: ty, to: &context, in: &env)
        }
    }

    @discardableResult
    private func emitLiteral<T>(_ literal: Literal, ofType type: Type,
                                to context: inout LLGenContext<T>,
                                in env: inout LLGenEnvironment) -> LLUnit {
        switch literal {
        case let .scalar(lit):
            switch lit {
            case let .bool(v):
                return v.constant
            case let .int(v):
                let llType = type.emit(to: &context, in: &env)
                return v ~ llType
            case let .float(v):
                let llType = type.emit(to: &context, in: &env)
                return v ~ llType
            }

        case let .array(xx):
            let vals = xx.map{$0.emit(to: &context, in: &env)}
            return vals.array(ofType: type.emit(to: &context, in: &env))

        case let .tuple(xx):
            return <%>xx.map{$0.emit(to: &context, in: &env)}

        case let .tensor(xx):
            let llType = type.emit(to: &context, in: &env)
            let elementTy = LLVMGetElementType(llType) ?? DLImpossibleResult()
            return xx.map { $0.emit(to: &context, in: &env) }.array(ofType: elementTy)

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
            var elements: [LLVMValueRef?] = fields.map { $0.1.emit(to: &context, in: &env) }
            return LLVMConstStruct(&elements, UInt32(fields.count), .false)

        case .zero:
            /// - Note: There doesn't seem to be a way to emit LLVM's `zeroinitializer` with
            /// the C API, so we are doing a `bitcast`.
            let llType = type.emit(to: &context, in: &env)
            let size = LLVMSizeOf(llType)
            let zero = LLVMConstInt(size, 0, .false)
            return LLVMConstBitCast(zero, llType)

        case .undefined:
            return LLVMGetUndef(type.emit(to: &context, in: &env))

        case .null:
            return LLVMConstNull(type.emit(to: &context, in: &env))
        }
    }
}

extension DLVM.Variable: LLEmittable {
    typealias LLUnit = LLVMValueRef
    @discardableResult
    func emit<T>(to context: inout LLGenContext<T>,
              in env: inout LLGenEnvironment) -> LLVMValueRef {
        let llType = type.emit(to: &context, in: &env)
        return LLVMAddGlobal(context.module, llType, name)
    }
}

extension DLVM.Module : LLEmittable {
    typealias LLUnit = LLVMModuleRef
    @discardableResult
    func emit<T>(to context: inout LLGenContext<T>,
              in env: inout LLGenEnvironment) -> LLUnit {
        for global in variables {
            global.emit(to: &context, in: &env)
        }
        for alias in typeAliases {
            alias.emit(to: &context, in: &env)
        }
        for fun in self {
            fun.emit(to: &context, in: &env)
        }
        return context.module
    }
}

extension DLVM.Function : LLEmittable {
    typealias LLUnit = LLVMValueRef
    @discardableResult
    func emit<T>(to context: inout LLGenContext<T>,
              in env: inout LLGenEnvironment) -> LLVMValueRef {
        DLUnimplemented()
    }
}

extension DLVM.BasicBlock : LLEmittable {
    typealias LLUnit = LLVMBasicBlockRef
    func emit<T>(to context: inout LLGenContext<T>,
              in env: inout LLGenEnvironment) -> LLVMBasicBlockRef {
        DLUnimplemented()
    }
}

extension DLVM.Instruction : LLEmittable {
    typealias LLUnit = LLVMValueRef
    @discardableResult
    func emit<T>(to context: inout LLGenContext<T>,
              in env: inout LLGenEnvironment) -> LLVMValueRef {
        let inst = kind.emit(to: &context, in: &env)
        LLVMSetValueName(inst, name)
        return inst
    }
}

extension DLVM.InstructionKind : LLEmittable {
    typealias LLUnit = LLVMValueRef
    @discardableResult
    func emit<T>(to context: inout LLGenContext<T>,
              in env: inout LLGenEnvironment) -> LLVMValueRef {
        let builder = context.builder
        switch self {
        case .allocateBox(_):
            DLUnimplemented()
        case .allocateHeap(let ty, count: let n):
            return LLVMBuildArrayMalloc(
                builder, ty.emit(to: &context, in: &env),
                n.emit(to: &context, in: &env), nil)
        case .allocateStack(let ty, let n):
            return LLVMBuildArrayAlloca(
                builder, ty.emit(to: &context, in: &env), %n, nil)
        case .apply(_, _):
            DLUnimplemented()
        case .bitCast(let v, let targetTy):
            return LLVMBuildBitCast(
                builder, v.emit(to: &context, in: &env),
                targetTy.emit(to: &context, in: &env), nil)
        case .branch(let bb, _):
            return LLVMBuildBr(builder, env.basicBlock(for: bb))
        case let .conditional(cond, thenBB, _, elseBB, _):
            return LLVMBuildCondBr(
                builder, cond.emit(to: &context, in: &env),
                env.basicBlock(for: thenBB), env.basicBlock(for: elseBB))
        case let .copy(from: src, to: dst, count: n):
            let srcVal = src.emit(to: &context, in: &env)
            let dstVal = dst.emit(to: &context, in: &env)
            let count = n.emit(to: &context, in: &env)
            let prototype = Builtin.Memory.memcpy(to: dstVal, from: srcVal, count: count,
                                                  align: 4 ~ i32, isVolatile: %false)
            return context.builtin.emit(prototype, using: builder)
        default:
            DLUnimplemented()
        }
    }
}
