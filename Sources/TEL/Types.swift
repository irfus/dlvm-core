//
//  Types.swift
//  DLVM
//
//  Created by Richard Wei on 1/21/17.
//
//

import struct DLVM.TensorShape

public enum FunctionTypeError : Error {
    case argumentCountMismatch(expected: Int, actual: Int)
    case shapeMismatch(TensorShape, TensorShape)
}

extension FunctionTypeError : CustomStringConvertible {
    public var description: String {
        switch self {

        case let .shapeMismatch(lhs, rhs):
            return "Expected two arguments of the same shape, but \(lhs) ≠ \(rhs)"

        case let .argumentCountMismatch(expected: expected, actual: actual):
            return "Expected \(expected) arguments, but \(actual) are provided"

        }
    }
}

/// - Note: This is not a formal type system. Type computation is purely defined
/// in the host language. I have sketched a somewhat formal type system in the 
/// previous commits but they are obviously extra work since we are not going
/// to allow function declarations in TEL.

public enum FunctionType {
    case monomorphicUnary, monomorphicBinary
    case matrixMultiplication, tensorMultiplication
    case comparison
}

internal extension FunctionType {
    func checkSanity(forArguments args: [TensorShape]) throws {
        guard args.count == argumentCount else {
            throw FunctionTypeError.argumentCountMismatch(expected: argumentCount, actual: args.count)
        }
    }
}

public extension FunctionType {
    public var argumentCount: Int {
        switch self {
        case .monomorphicUnary: return 1
        case .monomorphicBinary: return 2
        case .matrixMultiplication: return 2
        case .tensorMultiplication: return 2
        case .comparison: return 2
        }
    }

    private func broadcast(_ lhs: TensorShape, _ rhs: TensorShape) -> TensorShape? {
        return lhs.broadcasted(to: rhs) ?? rhs.broadcasted(to: lhs)
    }

    public func resultShape(forArguments args: [TensorShape]) throws -> TensorShape {
        try checkSanity(forArguments: args)
        switch self {
        case .monomorphicUnary: return args[0]
        case .monomorphicBinary, .comparison, .comparison:
            guard let broadcastedShape = broadcast(args[0], args[1]) else {
                throw FunctionTypeError.shapeMismatch(args[0], args[1])
            }
            return broadcastedShape
        case .matrixMultiplication:
            guard let resultShape = args[0].matrixMultiplied(with: args[1]) else {
                throw FunctionTypeError.shapeMismatch(args[0], args[1])
            }
            return resultShape
        case .tensorMultiplication:
            guard let resultShape = args[0].multiplied(with: args[1]) else {
                throw FunctionTypeError.shapeMismatch(args[0], args[1])
            }
            return resultShape
        }
    }
}
