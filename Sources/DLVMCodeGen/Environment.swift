//
//  Environment.swift
//  DLVM
//
//  Created by Richard Wei on 2/6/17.
//
//

import DLVM
import LLVM

public protocol LLEmittable {
    associatedtype LLUnit
    @discardableResult func emit<T :LLTarget>(to context: inout LLGenContext<T>,
                                              in env: inout LLGenEnvironment) -> LLUnit
}

public struct LLGenEnvironment {
    fileprivate var globals: [AnyHashable : IRValue] = [:]
    fileprivate var locals: [AnyHashable : IRValue] = [:]
    fileprivate var types: [TypeAlias : IRType] = [:]
}

extension LLGenEnvironment {
    mutating func clearLocals() {
        locals.removeAll()
    }

    mutating func insertGlobal<T : Definition & Hashable>
        (_ value: IRValue, for dlValue: T) {
        globals[dlValue] = value
    }

    mutating func insertLocal<T : Definition & Hashable>
        (_ value: IRValue, for dlValue: T) {
        locals[dlValue] = value
    }

    mutating func insertType(_ type: IRType, for typeAlias: TypeAlias) {
        types[typeAlias] = type
    }

    func value<T : Definition & Hashable>(for value: T) -> IRValue {
        return locals[value] ?? globals[value] ?? DLImpossibleResult()
    }

    func type(for alias: DLVM.TypeAlias) -> IRType {
        return types[alias] ?? DLImpossibleResult()
    }
}

public struct LLGenContext<TargetType :LLTarget> {
    public let dlModule: DLVM.Module
    public let context: LLVM.Context = Context.global
    public private(set) lazy var module: LLVM.Module = Module(name: self.dlModule.name, context: self.context)
    public private(set) lazy var target: TargetType = TargetType(module: self.module)
    public private(set) lazy var builder: LLVM.IRBuilder = IRBuilder(module: self.module)
    public init(module: DLVM.Module) {
        self.dlModule = module
    }
}
