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
import LLVM

public class CodeGenerator<TargetType : LLComputeTarget> {
    public let dlModule: DLVM.Module
    public lazy internal(set) var context: LLGenContext<TargetType> = LLGenContext(module: self.dlModule)
    public internal(set) var environment: LLGenEnvironment = LLGenEnvironment()

    public init(module: DLVM.Module) {
        self.dlModule = module
        do { try module.verify() }
        catch { DLImpossible() }
    }
}

extension DLVM.Use : LLEmittable {
    public typealias LLUnit = IRValue
    @discardableResult
    public func emit<T : LLComputeTarget>(to context: inout LLGenContext<T>,
                                   in env: inout LLGenEnvironment) -> IRValue {
        switch self {
        case let .global(_, val): return env.value(for: val)
        case let .function(fun): return env.value(for: fun)
        case let .argument(_, arg): return env.value(for: arg)
        case let .instruction(_, inst): return env.value(for: inst)
        case let .literal(litVal): return litVal.emit(to: &context, in: &env)
        }
    }
}

// MARK: - Literal lowering
extension DLVM.LiteralValue : LLEmittable {
    public typealias LLUnit = IRValue
    @discardableResult
    public func emit<T : LLComputeTarget>(to context: inout LLGenContext<T>,
                                   in env: inout LLGenEnvironment) -> LLUnit {
        switch literal {
        case let .scalar(lit):
            switch lit {
            case let .bool(v):
                return v.constant
            case let .int(v):
                let llType = type.emit(to: &context, in: &env) as? IntType ?? DLImpossibleResult()
                return llType.constant(v)
            case let .float(v):
                let llType = type.emit(to: &context, in: &env) as? FloatType ?? DLImpossibleResult()
                return llType.constant(v)
            }

        case let .array(xx):
            let vals = xx.map{$0.emit(to: &context, in: &env)}
            return ArrayType.constant(vals, type: xx[0].type.emit(to: &context, in: &env))

        case let .tuple(xx):
            let vals = xx.map{$0.emit(to: &context, in: &env)}
            return StructType.constant(values: vals)

        case let .struct(xx, isPacked: isPacked):
            let vals = xx.map{$0.emit(to: &context, in: &env)}
            return StructType.constant(values: vals, isPacked: isPacked)

        case let .tensor(xx):
            let vals = xx.map{$0.emit(to: &context, in: &env)}
            return ArrayType.constant(vals, type: xx[0].type.emit(to: &context, in: &env))

        case let .constant(constOp):
            return constOp.emit(to: &context, in: &env)

        case .zero:
            return type.emit(to: &context, in: &env).null()

        case .undefined:
            return type.emit(to: &context, in: &env).undef()

        case .null:
            return type.emit(to: &context, in: &env).null()
        }
    }
}

extension DLVM.TypeAlias : LLEmittable {
    public typealias LLUnit = IRType
    @discardableResult
    public func emit<T : LLComputeTarget>(to context: inout LLGenContext<T>,
                                   in env: inout LLGenEnvironment) -> IRType {
        let resultTy: IRType
        switch self {
        case let .opaque(name):
            return context.builder.createStruct(name: name)
        case let .transparent(name, .tuple(tt)):
            let llTypes = tt.map { $0.emit(to: &context, in: &env) }
            resultTy = context.builder.createStruct(name: name,
                                                    types: llTypes,
                                                    isPacked: true)
        case let .transparent(name, ty):
            let llType = ty.emit(to: &context, in: &env)
            resultTy = context.builder.createStruct(name: name,
                                                    types: [llType],
                                                    isPacked: false)
        }
        env.insertType(resultTy, for: self)
        return resultTy
    }
}

public extension DLVM.MemoryType {
    var addressSpace: LLAddressSpace {
        switch self {
        case .compute: return .global
        case .normal: return .generic
        }
    }
}

extension DLVM.`Type` : LLEmittable {
    public typealias LLUnit = IRType
    @discardableResult
    public func emit<T : LLComputeTarget>(to context: inout LLGenContext<T>,
                                   in env: inout LLGenEnvironment) -> IRType {
        switch self {
        case let .tensor(shape, dt):
            return shape.reduce(dt.llType, { acc, dim in
                ArrayType(elementType: acc, count: dim)
            })
        case .void:
            return VoidType()
        case .invalid:
            return VoidType()
        case let .tuple(subtt):
            return StructType(elementTypes: subtt.map{$0.emit(to: &context, in: &env)})
        case let .struct(structTy):
            return StructType(elementTypes: structTy.fields.map{$0.type.emit(to: &context, in: &env)},
                              isPacked: structTy.isPacked)
        case let .array(subt, n):
            return ArrayType(elementType: subt.emit(to: &context, in: &env), count: n)
        case let .function(args, ret):
            return FunctionType(argTypes: args.map{$0.emit(to: &context, in: &env)},
                                returnType: ret.emit(to: &context, in: &env))
        case let .alias(alias):
            return env.type(for: alias)
        case let .box(subt, .normal):
            return PointerType(pointee:
                StructType(elementTypes: [
                    context.referenceCounterType,
                    subt.emit(to: &context, in: &env) /// Direct storage
                ])
            )
        case let .box(subt, .compute):
            return PointerType(pointee:
                StructType(elementTypes: [
                    context.referenceCounterType,
                    PointerType(pointee: subt.emit(to: &context, in: &env),
                                addressSpace: .global) /// Indirect storage
                ])
            )
        case let .pointer(subt):
            return PointerType(pointee: subt.emit(to: &context, in: &env))
        case let .computeGraph(fun):
            return context.target.loweredComputeGraphType(from: fun)
        }
    }
}

extension DLVM.GlobalValue : LLEmittable {
    public typealias LLUnit = Global
    @discardableResult
    public func emit<T : LLComputeTarget>(to context: inout LLGenContext<T>,
                                   in env: inout LLGenEnvironment) -> Global {
        let initial = initializer.emit(to: &context, in: &env)
        let val = context.builder.addGlobal(name, initializer: initial)
        env.insertGlobal(val, for: self)
        return val
    }
}

extension DLVM.Module : LLEmittable {
    public typealias LLUnit = LLVM.Module
    @discardableResult
    public func emit<T : LLComputeTarget>(to context: inout LLGenContext<T>,
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
    public typealias LLUnit = LLVM.Function
    @discardableResult
    public func emit<T>(to context: inout LLGenContext<T>, in env: inout LLGenEnvironment) -> LLVM.Function where T : LLComputeTarget {
        /// Target handles compute functions
        if isCompute {
            return context.target.emitComputeFunction(from: self, to: &context, in: &env)
        }

        /// Non-compute functions are almost transparent to LLVM
        DLUnimplemented()
    }
}

extension DLVM.BasicBlock : LLEmittable {
    public typealias LLUnit = LLVM.BasicBlock
    public func emit<T>(to context: inout LLGenContext<T>, in env: inout LLGenEnvironment) -> LLVM.BasicBlock where T : LLComputeTarget {
        DLUnimplemented()
    }
}

extension DLVM.Instruction : LLEmittable {
    public typealias LLUnit = LLVM.Instruction
    @discardableResult
    public func emit<T>(to context: inout LLGenContext<T>, in env: inout LLGenEnvironment) -> LLVM.Instruction where T : LLComputeTarget {
        var inst = kind.emit(to: &context, in: &env)
        inst.name =? name
        return inst
    }
}

extension DLVM.InstructionKind : LLEmittable {
    public typealias LLUnit = LLVM.Instruction
    @discardableResult
    public func emit<T>(to context: inout LLGenContext<T>, in env: inout LLGenEnvironment) -> LLVM.Instruction where T : LLComputeTarget {
        DLUnimplemented()
    }
}

extension CodeGenerator {
    public func emit() -> LLVM.Module {
        return dlModule.emit(to: &context, in: &environment)
    }
}

public extension CodeGenerator {
    func writeBitcode(toFile file: String) throws {
        try context.module.emitBitCode(to: file)
    }
}
