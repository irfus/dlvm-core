//
//  Types.swift
//  DLVM
//
//  Created by Richard Wei on 1/21/17.
//
//

import struct DLVM.TensorShape

public enum TypeError : Error {
    case noArguments
    case argumentCountMismatch(actual: Int, expected: Int)
}

/// - Note: This is not a formal type system. Type computation is purely defined
/// in the host language. I have sketched a somewhat formal type system in the 
/// previous commits but they are obviously extra work since we are not going
/// to allow function declarations in TEL.

public class FunctionType {
    public var argumentCount: Int

    public init(argumentCount: Int) {
        self.argumentCount = argumentCount
    }

    internal func result(forArguments args: [TensorShape]) throws -> TensorShape {
        guard args.count == argumentCount else {
            throw TypeError.argumentCountMismatch(actual: args.count, expected: argumentCount)
        }
        guard let firstArg = args.first else {
            throw TypeError.noArguments
        }
        return firstArg
    }
}

public struct Function {
    public var name: String
    public var type: FunctionType

    public init(name: String, type: FunctionType) {
        self.name = name
        self.type = type
    }
}
