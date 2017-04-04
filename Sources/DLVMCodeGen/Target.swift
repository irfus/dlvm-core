//
//  Target.swift
//  DLVM
//
//  Created by Richard Wei on 3/21/17.
//
//

import LLVM

public protocol LLFunctionPrototype: Hashable {
    var name: StaticString { get }
    var type: FunctionType { get }
    var arguments: [IRValue] { get }
}

public protocol LLTypePrototype: Hashable {
    var name: StaticString { get }
    var type: IRType { get }
}

public extension LLFunctionPrototype {
    static func == (lhs: Self, rhs: Self) -> Bool {
        return lhs.name.utf8Start == rhs.name.utf8Start
    }
    
    var hashValue: Int {
        return name.utf8Start.hashValue
    }
}

public extension LLFunctionPrototype where Self : RawRepresentable, Self.RawValue == StaticString {
    var name: StaticString {
        return rawValue
    }
}

public extension LLTypePrototype where Self : RawRepresentable, Self.RawValue == StaticString {
    var name: StaticString {
        return rawValue
    }

    // Opaque by default
    public var type: IRType {
        return StructType(name: name.description)
    }
}

import DLVM

public protocol LLTarget : LLFunctionPrototypeCacheable {
    unowned var module: LLVM.Module { get }
    init(module: LLVM.Module)
}

public protocol LLComputeTarget : LLTarget {
    func loweredComputeGraphType(from function: DLVM.Function) -> StructType
    func emitComputeFunction(from function: DLVM.Function,
                             to context: inout LLGenContext<Self>,
                             in env: inout LLGenEnvironment) -> LLVM.Function
}

public protocol LLFunctionPrototypeCacheable : class {
    var functions: [AnyHashable : LLVM.Function] { get set }
    func function<T : LLFunctionPrototype>(from prototype: T) -> LLVM.Function
}

extension StaticString : Equatable {
    public static func == (lhs: StaticString, rhs: StaticString) -> Bool {
        return lhs.utf8Start == rhs.utf8Start
    }
}

extension LLTarget where Self : LLFunctionPrototypeCacheable {
    func build<T : LLFunctionPrototype>(
               _ prototype: T,
               using builder: LLVM.IRBuilder,
               name: String = "") -> IRValue {
        let function = self.function(from: prototype)
        return builder.buildCall(function, args: prototype.arguments, name: name)
    }
}

extension LLFunctionPrototypeCacheable where Self : LLTarget {
    public func function<T : LLFunctionPrototype>(from prototype: T) -> LLVM.Function {
        if let fun = functions[prototype] {
            return fun
        }
        let builder = IRBuilder(module: module)
        let fun = builder.addFunction(prototype.name.description,
                                      type: prototype.type)
        functions[prototype] = fun
        return fun
    }
}
