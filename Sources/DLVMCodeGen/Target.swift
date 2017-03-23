//
//  Target.swift
//  DLVM
//
//  Created by Richard Wei on 3/21/17.
//
//

import LLVM

public protocol FunctionPrototype : Hashable {
    var name: StaticString { get }
    var type: FunctionType { get }
    var arguments: [IRValue] { get }
}

public protocol TypePrototype : Hashable {
    var name: StaticString { get }
    var type: IRType { get }
}

public extension FunctionPrototype {
    static func == (lhs: Self, rhs: Self) -> Bool {
        return lhs.name.utf8Start == rhs.name.utf8Start
    }
    
    var hashValue: Int {
        return name.utf8Start.hashValue
    }
}

public extension FunctionPrototype where Self : RawRepresentable, Self.RawValue == StaticString {
    var name: StaticString {
        return rawValue
    }
}

public extension TypePrototype where Self : RawRepresentable, Self.RawValue == StaticString {
    var name: StaticString {
        return rawValue
    }

    // Opaque by default
    public var type: IRType {
        return StructType(name: name.description)
    }
}

public protocol Target {
    unowned var module: Module { get }
    init(module: Module)
}

public protocol FunctionPrototypeCache : class {
    var functions: [AnyHashable : Function] { get set }
    func function<T : FunctionPrototype>(from prototype: T) -> Function
}

extension StaticString : Equatable {
    public static func == (lhs: StaticString, rhs: StaticString) -> Bool {
        return lhs.utf8Start == rhs.utf8Start
    }
}


extension Target where Self : FunctionPrototypeCache {
    func build<T : FunctionPrototype>(
               _ prototype: T,
               using builder: IRBuilder,
               name: String = "") -> IRValue {
        let function = self.function(from: prototype)
        return builder.buildCall(function, args: prototype.arguments, name: name)
    }
}

extension FunctionPrototypeCache where Self : Target {
    public func function<T : FunctionPrototype>(from prototype: T) -> Function {
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
