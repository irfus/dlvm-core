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

public protocol LLTarget : LLFunctionPrototypeCacheable {
    unowned var module: Module { get }
    init(module: Module)
}

public protocol LLFunctionPrototypeCacheable : class {
    var functions: [AnyHashable : Function] { get set }
    func function<T : LLFunctionPrototype>(from prototype: T) -> Function
}

extension StaticString : Equatable {
    public static func == (lhs: StaticString, rhs: StaticString) -> Bool {
        return lhs.utf8Start == rhs.utf8Start
    }
}

extension LLTarget where Self : LLFunctionPrototypeCacheable {
    func build<T : LLFunctionPrototype>(
               _ prototype: T,
               using builder: IRBuilder,
               name: String = "") -> IRValue {
        let function = self.function(from: prototype)
        return builder.buildCall(function, args: prototype.arguments, name: name)
    }
}

extension LLFunctionPrototypeCacheable where Self : LLTarget {
    public func function<T : LLFunctionPrototype>(from prototype: T) -> Function {
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
