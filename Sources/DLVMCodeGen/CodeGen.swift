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

// MARK: - Use lowering

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

// MARK: - IR unit lowering

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

// MARK: - Instruction Lowering

extension DLVM.Instruction : LLEmittable {
    typealias LLUnit = LLVMValueRef
    @discardableResult
    func emit<T>(to context: inout LLGenContext<T>,
              in env: inout LLGenEnvironment) -> LLVMValueRef {
        let inst = kind.emit(to: &context, in: &env, basicBlock: parent)
        LLVMSetValueName(inst, name)
        return inst
    }
}

extension DLVM.InstructionKind {
    typealias LLUnit = LLVMValueRef
    @discardableResult
    func emit<T>(to context: inout LLGenContext<T>,
              in env: inout LLGenEnvironment, basicBlock: BasicBlock) -> LLVMValueRef {
        let builder = context.builder
        let function = basicBlock.parent
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

        case .dataTypeCast(_, _):
            /// Emit dtype-cast kernel or LLVM conversion intrinsic
            DLUnimplemented()

        case let .deallocate(v):
            let val = v.emit(to: &context, in: &env)
            let prototype = Builtin.Memory.free(val)
            return context.builtin.emit(prototype, using: builder)

        case let .elementPointer(v, kk):
            let val = v.emit(to: &context, in: &env)
            var gepPath = v.type.emitIndexPath(for: kk, to: &context, in: &env) as [LLVMValueRef?]
            return LLVMBuildGEP(builder, val, &gepPath, UInt32(gepPath.count), nil)

        case let .extract(from: v, at: kk):
            let val = v.emit(to: &context, in: &env)
            /// Since we need to handle indirect passing, we have two cases
            /// If indirectly passed, use GEP to get the pointer to the element
            if v.type.shouldBePassedIndirectly {
                var path = v.type.emitIndexPath(for: kk, to: &context, in: &env) as [LLVMValueRef?]
                return LLVMBuildGEP(builder, val, &path, UInt32(path.count), nil)
            }
            /// Otherwise (directly passed), emit `extractvalue` for aggregate values
            /// or `extractelement` for vectors
            switch v.type {
            /// If a tensor were passed directly, it's definitely passed as an LLVM vector,
            /// and thus we should cast it to restore the aggregate structure and emit an
            /// `extractvalue` for each index, and finally cast that to an LLVM vector if needed
            case let .tensor(shape, dtype):
                let indices = v.type.staticIndexPath(for: kk) ?? DLImpossibleResult()
                /// Assert that the number of indices is legal
                DLAssert(indices.count == shape.rank)
                /// If it's a rank-1 tensor (literally a vector), omit bitcast and
                /// emit `extractelement` directly
                if shape.isVector {
                    return LLVMBuildExtractElement(builder, val, %indices[0], nil)
                }
                /// Otherwise, bitcast is required so that we can reuse LLVM's indexing arithmetics
                let array = LLVMBuildBitCast(builder, val, shape.loweredArrayType(of: dtype), nil) ?? DLImpossibleResult()
                let element = indices.reduce(array) { newVal, idx in
                    LLVMBuildExtractValue(builder, newVal, idx, nil)!
                }
                /// Cast it back to vector if needed
                let elementShape = shape.dropFirst(indices.count)
                if elementShape.rank > 0 {
                    return LLVMBuildBitCast(builder, element,
                                            elementShape.loweredVectorType(of: dtype), nil)!
                }
                return element

            /// Otherwise it's a non-vector aggregate value in LLVM, for which we emit `extractvalue`
            default:
                /// Get static indices and build an `extractvalue` instruction for each index
                let indices = v.type.staticIndexPath(for: kk) ?? DLImpossibleResult()
                var currentVal = val
                for index in indices {
                    currentVal = LLVMBuildExtractValue(builder, currentVal, index, nil)
                }
                return currentVal
            }

        case let .insert(src, to: dest, at: kk):
            let srcVal = src.emit(to: &context, in: &env)
            var destVal = dest.emit(to: &context, in: &env)
            if dest.type.shouldBePassedIndirectly {
                let users = try! function.analysis(from: UserAnalysis.self)
                /// If `dest` has users other than the current instruction, emit a copy
                /// TODO: check for other users' data locality requirements to avoid
                /// copying as much as possible
                if let destDef = dest.value as? Definition, users[destDef].count > 1 {
                    /// TODO: work with memory tracker
                }

            }
            DLUnimplemented()


        default:
            DLUnimplemented()
        }
    }
}
