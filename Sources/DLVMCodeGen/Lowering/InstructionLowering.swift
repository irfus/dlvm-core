//
//  InstructionLowering.swift
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

extension DLVM.Instruction : LLEmittable {
    public typealias LLUnit = LLVMValueRef
    @discardableResult
    public func emit<T>(to context: LLGenContext<T>,
              in env: LLGenEnvironment) -> LLVMValueRef {
        let inst = kind.emit(to: context, in: env, basicBlock: parent)
        LLVMSetValueName(inst, name)
        return inst
    }
}

extension DLVM.InstructionKind {
    public typealias LLUnit = LLVMValueRef
    @discardableResult
    public func emit<T>(to context: LLGenContext<T>, in env: LLGenEnvironment,
                     basicBlock: BasicBlock) -> LLVMValueRef {
        let builder = context.builder
        let function = basicBlock.parent
        switch self {
        case .allocateBox(_):
            DLUnimplemented()

        case .allocateHeap(let ty, count: let n):
            let ptr = LLVMBuildArrayMalloc(
                builder, ty.emit(to: context, in: env),
                n.emit(to: context, in: env), nil) ?? DLImpossibleResult()
            let size = ty.size(for: Builtin.self) ?? DLImpossibleResult()
            context.builtin.emit(Builtin.Memory.lifetimeStart(
                size: %size,
                pointer: ptr), using: builder)
            return ptr

        case .allocateStack(let ty, let n):
            let llType = ty.emit(to: context, in: env)
            let ptr = LLVMBuildArrayAlloca(builder, llType, %n, nil) ?? DLImpossibleResult()
            let size = ty.size(for: Builtin.self) ?? DLImpossibleResult()
            context.builtin.emit(Builtin.Memory.lifetimeStart(size: %size, pointer: ptr), using: builder)
            return ptr

        case let .apply(.function(ty, fn), args):
            /// - TODO: Deal with indirect passing
            DLUnimplemented()

        case let .apply(use, args):
            let fnVal = use.emit(to: context, in: env)
            var argVals = args.map { arg in
                arg.emit(to: context, in: env) as LLVMValueRef?
            }
            return LLVMBuildCall(builder, fnVal, &argVals, UInt32(argVals.count), nil)

        case .bitCast(let v, let targetTy):
            return LLVMBuildBitCast(
                builder, v.emit(to: context, in: env),
                targetTy.emit(to: context, in: env), nil)

        case .branch(let bb, _):
            return LLVMBuildBr(builder, env.basicBlock(for: bb))

        case let .conditional(cond, thenBB, _, elseBB, _):
            return LLVMBuildCondBr(
                builder, cond.emit(to: context, in: env),
                env.basicBlock(for: thenBB), env.basicBlock(for: elseBB))

        case let .copy(from: src, to: dst, count: n):
            let srcVal = src.emit(to: context, in: env)
            let dstVal = dst.emit(to: context, in: env)
            let count = n.emit(to: context, in: env)
            let prototype = Builtin.Memory.memcpy(to: dstVal, from: srcVal,
                                                  count: count,
                                                  align: 4.llValue(ofType: i32),
                                                  isVolatile: %false)
            return context.builtin.emit(prototype, using: builder)

        case .dataTypeCast(_, _):
            /// Emit dtype-cast kernel or LLVM conversion intrinsic
            DLUnimplemented()

        case let .deallocate(v):
            DLAssert(v.type.isPointer)
            let val = v.emit(to: context, in: env)
            let prototype = Builtin.Memory.free(val)
            return context.builtin.emit(prototype, using: builder)

        case let .elementPointer(v, kk):
            let val = v.emit(to: context, in: env)
            var gepPath = v.type.emitIndexPath(for: kk, to: context, in: env) as [LLVMValueRef?]
            return LLVMBuildGEP(builder, val, &gepPath, UInt32(gepPath.count), nil)

        case let .extract(from: v, at: kk):
            let val = v.emit(to: context, in: env)
            /// Since we need to handle indirect passing, we have two cases
            /// If indirectly passed, use GEP to get the pointer to the element
            if v.type.shouldBePassedIndirectly {
                var path = v.type.emitIndexPath(for: kk, to: context, in: env) as [LLVMValueRef?]
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
                let array = LLVMBuildBitCast(builder, val, shape.loweredArrayType(of: dtype), nil)
                    ?? DLImpossibleResult()
                let element: LLVMValueRef = indices.reduce(array) { newVal, idx in
                    LLVMBuildExtractValue(builder, newVal, idx, nil) ?? DLImpossibleResult()
                }
                /// Cast it back to vector if needed
                let elementShape = shape.dropFirst(indices.count)
                if elementShape.rank > 0 {
                    return LLVMBuildBitCast(builder, element,
                                            elementShape.loweredVectorType(of: dtype), nil) ?? DLImpossibleResult()
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

        case let .insert(src, to: dest, at: keys):
            let srcVal = src.emit(to: context, in: env)
            let destVal = dest.emit(to: context, in: env)
            /// Deal with memory if indirect
            if dest.type.shouldBePassedIndirectly {
                let users = try! function.analysis(from: UserAnalysis.self)
                /// If `dest` has users other than the current instruction, emit a copy
                /// TODO: check for other users' data locality requirements to avoid
                /// copying as much as possible
                if let destDef = dest.value as? Definition, users[destDef].count > 1 {
                    /// TODO: work with memory tracker
                    DLUnimplemented()
                }
            }
            /// Otherwise, lower to LLVM's insertion instructions
            switch dest.type {
            case let .tensor(shape, dtype):
                var indices = dest.type.staticIndexPath(for: keys) ?? DLImpossibleResult()
                /// Assert that the number of indices is legal
                DLAssert(indices.count == shape.rank)
                /// If it's a rank-1 tensor (literally a vector), omit bitcast and
                /// emit `extractelement` directly
                if shape.isVector {
                    return LLVMBuildInsertElement(builder, destVal, srcVal, %indices[0], nil)
                }
                /// Otherwise, bitcast is required so that we can reuse LLVM's indexing arithmetics
                let array = LLVMBuildBitCast(builder, destVal, shape.loweredArrayType(of: dtype), nil)
                    ?? DLImpossibleResult()
                /// Cast the element's type if necessary
                guard case let .tensor(srcShape, srcDType) = src.type else { DLImpossible() }
                let elemVal: LLVMValueRef
                if srcShape.isScalar {
                    elemVal = srcVal
                } else {
                    elemVal = LLVMBuildBitCast(builder, srcVal, srcShape.loweredArrayType(of: srcDType), nil)
                }
                /// Insert element to the high-dimensional array at indices
                let result = LLVMConstInsertValue(array, elemVal, &indices, UInt32(indices.count)) ?? DLImpossibleResult()
                /// Cast it back to vector if needed
                if shape.isScalar {
                    return result
                }
                return LLVMBuildBitCast(builder, result, shape.loweredVectorType(of: dtype), nil)

            /// Otherwise it's a non-vector aggregate value in LLVM, for which we emit `extractvalue`
            default:
                /// Get static indices and build an `extractvalue` instruction for each index
                var indices = dest.type.staticIndexPath(for: keys) ?? DLImpossibleResult()
                guard case let .tensor(srcShape, srcDType) = src.type else { DLImpossible() }
                let elemVal: LLVMValueRef
                if srcShape.isScalar {
                    elemVal = srcVal
                } else {
                    elemVal = LLVMBuildBitCast(builder, srcVal, srcShape.loweredArrayType(of: srcDType), nil)
                }
                return LLVMConstInsertValue(destVal, elemVal, &indices, UInt32(indices.count))
            }

        case let .load(use):
            DLAssert(use.type.isPointer)
            let val = use.emit(to: context, in: env)
            if use.type.shouldBePassedIndirectly {
                return val
            }
            return LLVMBuildLoad(builder, val, nil)

        case let .map(op, use):
            return emitMap(op, use, to: context, in: env)

        case let .zipWith(op, lhs, rhs):
            return emitZipWith(op, lhs, rhs, to: context, in: env)

        case let .matrixMultiply(lhs, rhs):
            return emitMatrixMultiply(lhs, rhs, to: context, in: env)

            // case let .concatenate(args, axis: axis):

        case let .release(use):
            let val = use.emit(to: context, in: env)
            let proto = Builtin.Reference.release(val)
            return context.builtin.emit(proto, using: builder)

        case let .retain(use):
            let val = use.emit(to: context, in: env)
            let proto = Builtin.Reference.retain(val)
            return context.builtin.emit(proto, using: builder)

        case let .store(src, to: dest):
            DLAssert(dest.type.isPointer(to: src.type))
            let srcVal = src.emit(to: context, in: env)
            let destVal = dest.emit(to: context, in: env)
            /// If indirect, [TODO] request memory, and memcpy
            if src.type.shouldBePassedIndirectly {
                /// - TODO: Request memory
                let llType = LLVMTypeOf(destVal)
                let proto = Builtin.Memory.memcpy(to: destVal, from: srcVal, count: LLVMSizeOf(llType), align: LLVMAlignOf(llType), isVolatile: %true)
                return context.builtin.emit(proto, using: builder)
            }
            /// Otherwise, emit a normal `store`
            return LLVMBuildStore(builder, srcVal, destVal)

        case let .transpose(use):
            /// - TODO: Launch transpose kernel
            return emitTranspose(use, to: context, in: env)

        case .trap:
            let proto = Builtin.Control.trap
            return context.builtin.emit(proto, using: builder)

        default:
            DLUnimplemented()
        }
    }

    func emitMap<T>(_ operator: UnaryOp, _ argument: Use,
                 to context: LLGenContext<T>, in env: LLGenEnvironment) -> LLVMValueRef {
        DLUnimplemented()
    }

    func emitZipWith<T>(_ operator: BinaryOp, _ lhs: Use, _ rhs: Use,
                     to context: LLGenContext<T>, in env: LLGenEnvironment) -> LLVMValueRef {
        DLUnimplemented()
    }

    func emitMatrixMultiply<T>(_ lhs: Use, _ rhs: Use,
                            to context: LLGenContext<T>, in env: LLGenEnvironment) -> LLVMValueRef {
        DLUnimplemented()
    }

    func emitTranspose<T>(_ argument: Use,
                       to context: LLGenContext<T>, in env: LLGenEnvironment) ->  LLVMValueRef {
        DLUnimplemented()
    }
}
