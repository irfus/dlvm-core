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

public class CodeGenerator<TargetType : LLTarget> {
    public let dlModule: DLVM.Module
    public lazy internal(set) var context: LLGenContext<TargetType> = LLGenContext(module: self.dlModule)
    public internal(set) var environment: LLGenEnvironment = LLGenEnvironment()

    public init(module: DLVM.Module) {
        self.dlModule = module
        do { try module.verify() }
        catch { DLImpossible() }
    }
}

public enum LLGenError : Error {
    case fileError
}

public extension Type {
    func emitIndexPath<T>(from keyPath: [ElementKey],
                          to context: inout LLGenContext<T>,
                          in env: inout LLGenEnvironment) -> [LLVMValueRef] {
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
                indices.append(v.emit(to: &context, in: &env))
            default:
                DLImpossible()
            }
            current = current.subtype(at: key) ?? DLImpossibleResult()
        }
        return indices
    }
}

extension DLVM.Use : LLEmittable {
    public typealias LLUnit = LLVMValueRef
    @discardableResult
    public func emit<T>(to context: inout LLGenContext<T>,
                        in env: inout LLGenEnvironment) -> LLVMValueRef {
        switch self {
        case let .global(_, val):
            return env.value(for: val)
        case let .function(_, fun):
            return env.value(for: fun)
        case let .argument(_, arg):
            return env.value(for: arg)
        case let .instruction(_, inst):
            return env.value(for: inst)
        case let .literal(ty, lit):
            return emitLiteral(lit, ofType: ty, to: &context, in: &env)
        case .constant(_, _):
            DLUnimplemented()
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
            case .float(_):
                let llType = type.emit(to: &context, in: &env)

                DLUnimplemented()
                // return v ~ llType
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
            guard case let .struct(structTy) = type else { DLImpossible() }
            var elements: [LLVMValueRef?] = fields.map { $0.1.emit(to: &context, in: &env) }
            return LLVMConstStruct(&elements, UInt32(fields.count),
                                   structTy.isPacked ? .true : .false)

        case .zero:
            DLUnimplemented()

        case .undefined:
            return LLVMGetUndef(type.emit(to: &context, in: &env))

        case .null:
            return LLVMConstNull(type.emit(to: &context, in: &env))
        }
    }
}

extension DLVM.TypeAlias : LLEmittable {
    public typealias LLUnit = LLVMTypeRef
    @discardableResult
    public func emit<T>(to context: inout LLGenContext<T>,
                        in env: inout LLGenEnvironment) -> LLVMTypeRef {
        guard let type = type else {
            return ^name // opaque
        }
        return type.emit(to: &context, in: &env)
    }
}

extension DLVM.StructType : LLEmittable {
    public typealias LLUnit = LLVMTypeRef
    public func emit<T>(to context: inout LLGenContext<T>,
                        in env: inout LLGenEnvironment) -> LLVMTypeRef {

        var elements: [LLVMTypeRef?] = subtypes.map { $0.emit(to: &context, in: &env) }
        return LLVMStructType(&elements, UInt32(elements.count), isPacked ? .true : .false)
    }
}

extension DLVM.`Type` : LLEmittable {
    public typealias LLUnit = LLVMTypeRef
    @discardableResult
    public func emit<T>(to context: inout LLGenContext<T>,
                        in env: inout LLGenEnvironment) -> LLVMTypeRef {
        switch self {
        case let .tensor(shape, dt):
            return shape.reduce(dt.llType, { acc, dim in dim * acc })
        case .void:
            return LLVMVoidType()
        case .invalid:
            return LLVMVoidType()
        case let .tuple(subtt):
            return ^subtt.map{$0.emit(to: &context, in: &env)}
        case let .struct(structTy):
            return structTy.emit(to: &context, in: &env)
        case let .array(n, subt):
            return n * subt.emit(to: &context, in: &env)
        case let .function(args, ret):
            return args.map{$0.emit(to: &context, in: &env)} => ret.emit(to: &context, in: &env)
        case let .alias(alias):
            return env.type(for: alias)
        case let .box(subt):
            return ^[referenceCounterType, subt.emit(to: &context, in: &env)]
        case let .pointer(subt):
            return subt.emit(to: &context, in: &env)*
        }
    }
}

extension DLVM.GlobalValue : LLEmittable {
    public typealias LLUnit = LLVMValueRef
    @discardableResult
    public func emit<T>(to context: inout LLGenContext<T>,
                        in env: inout LLGenEnvironment) -> LLVMValueRef {
        DLUnimplemented()
    }
}

extension DLVM.Module : LLEmittable {
    public typealias LLUnit = LLVMModuleRef
    @discardableResult
    public func emit<T>(to context: inout LLGenContext<T>,
                        in env: inout LLGenEnvironment) -> LLUnit {
        for global in globalValues {
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
    public typealias LLUnit = LLVMValueRef
    @discardableResult
    public func emit<T>(to context: inout LLGenContext<T>,
                        in env: inout LLGenEnvironment) -> LLVMValueRef {
        DLUnimplemented()
    }
}

extension DLVM.BasicBlock : LLEmittable {
    public typealias LLUnit = LLVMBasicBlockRef
    public func emit<T>(to context: inout LLGenContext<T>,
                        in env: inout LLGenEnvironment) -> LLVMBasicBlockRef {
        DLUnimplemented()
    }
}

extension DLVM.Instruction : LLEmittable {
    public typealias LLUnit = LLVMValueRef
    @discardableResult
    public func emit<T>(to context: inout LLGenContext<T>,
                        in env: inout LLGenEnvironment) -> LLVMValueRef {
        let inst = kind.emit(to: &context, in: &env)
        LLVMSetValueName(inst, name)
        return inst
    }
}

extension DLVM.InstructionKind : LLEmittable {
    public typealias LLUnit = LLVMValueRef
    @discardableResult
    public func emit<T>(to context: inout LLGenContext<T>,
                        in env: inout LLGenEnvironment) -> LLVMValueRef {
        DLUnimplemented()
    }
}

extension CodeGenerator {
    public func emit() -> LLVMModuleRef {
        return dlModule.emit(to: &context, in: &environment)
    }
}

public extension CodeGenerator {
    func writeBitcode(toFile file: String) throws {
        guard LLVMWriteBitcodeToFile(context.module, file) == 0 else {
            throw LLGenError.fileError
        }
    }
}
