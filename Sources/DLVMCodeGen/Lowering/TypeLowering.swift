//
//  TypeLowering.swift
//  DLVMCodeGen
//
//  Copyright 2016-2017 The DLVM Team.
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
import struct CoreTensor.TensorShape

// MARK: - Type lowering

extension DLVM.TypeAlias : LLEmittable {
    public typealias LLUnit = LLVMTypeRef
    @discardableResult
    public func emit<T>(to context: LLGenContext<T>,
                     in env: LLGenEnvironment) -> LLVMTypeRef {
        guard let type = type else {
            return ^name // opaque
        }
        return type.emit(to: context, in: env)
    }
}

extension DLVM.StructType : LLEmittable {
    public typealias LLUnit = LLVMTypeRef
    public func emit<T>(to context: LLGenContext<T>,
                     in env: LLGenEnvironment) -> LLVMTypeRef {
        var elements: [LLVMTypeRef?] = elementTypes.map { $0.emit(to: context, in: env) }
        return LLVMStructType(&elements, UInt32(elements.count), .false)
    }
}

extension TensorShape {
    func loweredArrayType(of elementType: DataType) -> LLVMTypeRef {
        return reduce(elementType.llType, {$1 * $0})
    }

    func loweredVectorType(of elementType: DataType) -> LLVMTypeRef {
        return LLVMVectorType(elementType.llType, UInt32(contiguousSize))
    }
}

extension DLVM.`Type` : LLEmittable {
    public typealias LLUnit = LLVMTypeRef

    private func directlyEmit<T>(to context: LLGenContext<T>,
                              in env: LLGenEnvironment) -> LLVMTypeRef {
        switch self {
        case .invalid:
            DLImpossible()
        case let .tensor(shape, dt):
            return shape.loweredArrayType(of: dt)
        case .void:
            return LLVMVoidType()
        case .stack:
            return stackType
        case let .tuple(elemTypes):
            return ^elemTypes.map{$0.emit(to: context, in: env)}
        case let .struct(structTy):
            return env.type(for: structTy)
        case let .array(n, elemType):
            return n * elemType.emit(to: context, in: env)
        case let .function(args, ret):
            return args.map{$0.emit(to: context, in: env)}
                => ret.emit(to: context, in: env)
        case let .alias(alias):
            return env.type(for: alias)
        case let .box(boxeeType):
            return ^[referenceCounterType, boxeeType.emit(to: context, in: env)]
        case let .pointer(subt):
            return subt.emit(to: context, in: env)*
        }
    }

    @discardableResult
    public func emit<T>(to context: LLGenContext<T>,
                     in env: LLGenEnvironment) -> LLVMTypeRef {
        switch self {
        /// Function type with indirectly passed return value
        case let .function(args, ret) where ret.shouldBePassedIndirectly:
            /// If result should be passed indirectly, make it a pointer
            /// at the end of the parameter list
            var argTypes = args.map{$0.emit(to: context, in: env)}
            let retType = ret.directlyEmit(to: context, in: env)
            argTypes.append(retType*)
            return argTypes => LLVMVoidType()
        /// Directly passed tensors should be converted to LLVM vectors
        case let .tensor(shape, dtype) where !shouldBePassedIndirectly:
            return LLVMVectorType(dtype.llType, UInt32(shape.contiguousSize))
        /// For all indirectly passed types, emit a pointer to their bare type
        case _ where shouldBePassedIndirectly:
            let bareType = directlyEmit(to: context, in: env)
            return bareType*
        /// For all other types, directly emit them
        default:
            return directlyEmit(to: context, in: env)
        }
    }

    /// Determine if values of this type should be passed by reference
    var shouldBePassedIndirectly: Bool {
        switch canonical {
        case .alias(_): return false
        case .box(_): return false
        case .function(_, _): return false
        case .pointer(_): return false
        case .void: return false
        /// Struct
        case let .struct(structTy):
            return structTy.fields.forAll { (_, ty) in
                ty.shouldBePassedIndirectly
            }
        /// Scalar tensor
        case let .tensor(shape, _):
            /// - TODO: add bitwidth-based and target-specific calculation
            return shape.contiguousSize > 4
        /// Array
        case let .array(n, ty):
            /// - TODO: add bitwidth-based and target-specific calculation
            return n > 4 || ty.shouldBePassedIndirectly
        /// Tuple
        case let .tuple(elems):
            return elems.forAll { $0.shouldBePassedIndirectly }
        /// Stack
        case .stack:
            // Consider changing to true
            return false
        /// Impossible case
        case .invalid: DLImpossible()
        }
    }

    /// Emit an index path (a list of element keys) for LLVM GEP
    func emitIndexPath<T>(for keyPath: [ElementKey],
                       to context: LLGenContext<T>,
                       in env: LLGenEnvironment) -> [LLVMValueRef] {
        var current: Type = self
        var indices: [LLVMValueRef] = []
        for key in keyPath {
            switch (key, current) {
            case let (.index(i), _):
                indices.append(%i)
            case let (.name(n), .struct(structTy)):
                let index = structTy.indexOfField(named: n) ?? DLImpossibleResult()
                indices.append(%index)
            case let (.value(v), _):
                indices.append(v.emit(to: context, in: env))
            default:
                DLImpossible()
            }
            current = current.elementType(at: key) ?? DLImpossibleResult()
        }
        return indices
    }

    /// Get the static index path for LLVM `extractvalue`, which requires a
    /// static integer index
    func staticIndexPath(for keyPath: [ElementKey]) -> [UInt32]? {
        var current: Type = self
        var indices: [UInt32] = []
        for key in keyPath {
            switch (key, current) {
            case let (.index(i), _):
                indices.append(UInt32(i))
            case let (.name(n), .struct(structTy)):
                let index = structTy.indexOfField(named: n) ?? DLImpossibleResult()
                indices.append(UInt32(index))
            default:
                /// Value index is not considered static
                return nil
            }
            current = current.elementType(at: key) ?? DLImpossibleResult()
        }
        return indices
    }
}

extension StructType {
    private func nextOffset<Target : NativeTarget>(from offset: Int, for _: Target.Type) -> Int {
        let align = Target.pointerSize
        return (offset + align - 1) & ~(align - 1)
    }

    func size<Target : NativeTarget>(for target: Target.Type) -> Int? {
        var size = 0
        guard let last = fields.last else {
            return 1
        }
        for (_, type) in fields.dropLast() {
            guard let fieldSize = type.size(for: target) else {
                return nil
            }
            size = nextOffset(from: size + fieldSize, for: Target.self)
        }
        return last.type.size(for: target).flatMap { $0 + size }
    }
}

extension DataType {
    var size: Int {
        switch self {
        case .bool: return 1
        case let .float(size): return Int(size.rawValue)
        case let .int(size): return Int(size)
        }
    }
}

extension Type {
    func size<Target : NativeTarget>(for target: Target.Type) -> Int? {
        switch canonical {
        case .alias(_), .invalid, .void: return nil
        case let .array(n, ty):
            return ty.size(for: target).flatMap { $0 * n }
        case .box(_):
            return 2 * Target.pointerSize
        case .pointer(_):
            return Target.pointerSize
        case .function(_, _):
            return Target.pointerSize
        case .stack:
            return Target.pointerSize
        case let .struct(structTy):
            return structTy.size(for: target)
        case let .tensor(shape, dtype):
            return shape.contiguousSize * dtype.size
        case let .tuple(elemTypes):
            var size = 0
            for ty in elemTypes {
                guard let elemSize = ty.size(for: target) else {
                    return nil
                }
                size += elemSize
            }
            return size
        }
    }
}
