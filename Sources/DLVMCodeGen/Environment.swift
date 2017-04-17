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
import LLVM

public protocol LLEmittable {
    associatedtype LLUnit
    @discardableResult func emit<T : LLComputeTarget>(to context: inout LLGenContext<T>,
                                                      in env: inout LLGenEnvironment) -> LLUnit
}

/// Environment contains mappings from DLVM definitions to LLVM definitions
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

/// Context contains module, target, builder, etc
public struct LLGenContext<TargetType : LLTarget> {
    public let dlModule: DLVM.Module
    private let context: LLVM.Context = Context.global
    public private(set) lazy var module: LLVM.Module = Module(name: self.dlModule.name, context: self.context)
    public private(set) lazy var target: TargetType = TargetType(module: self.module)
    public private(set) lazy var builder: LLVM.IRBuilder = IRBuilder(module: self.module)
    public init(module: DLVM.Module) {
        self.dlModule = module
    }
}

