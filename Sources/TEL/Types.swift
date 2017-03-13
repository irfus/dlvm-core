//
//  Types.swift
//  DLVM
//
//  Created by Richard Wei on 1/21/17.
//
//

import DLVM

public enum FunctionTypeError : Error {
    case argumentCountMismatch(expected: Int, actual: Int)
    case shapeMismatch(TensorShape, TensorShape)
    case concatEmpty
    case notSupported
}

extension FunctionTypeError : CustomStringConvertible {
    public var description: String {
        switch self {

        case let .shapeMismatch(lhs, rhs):
            return "Expected two arguments of the same shape, but \(lhs) ≠ \(rhs)"

        case let .argumentCountMismatch(expected: expected, actual: actual):
            return "Expected \(expected) arguments, but \(actual) are provided"

        case .notSupported:
            return "Function not yet supported"

        case .concatEmpty:
            return "Expected at least one argument in concatenation"

        }
    }
}

extension DLVM.OpKind {

    private func checkArgumentCount(with arguments: [Use]) throws {
        guard argumentCount == arguments.count else {
            throw FunctionTypeError.argumentCountMismatch(expected: argumentCount,
                                                          actual: arguments.count)
        }
    }

    func makeOperation(with arguments: [Use]) throws -> Operation {
        try checkArgumentCount(with: arguments)
        switch self {
        case let .unary(op):
            return .unary(op, arguments[0])
        case let .binary(op):
            return .binary(op, arguments[0], arguments[1])
        case .matrixMultiply:
            return .matrixMultiply(arguments[0], arguments[1])
        default:
            throw FunctionTypeError.notSupported
        }
    }

    func resultShape(forArguments args: [TensorShape]) throws -> TensorShape {
        guard args.count == argumentCount else {
            throw FunctionTypeError.argumentCountMismatch(expected: argumentCount, actual: args.count)
        }
        switch self {
        case .concatenate where args.isEmpty:
            throw FunctionTypeError.concatEmpty
        case .concatenate:
            return try args.dropFirst().reduce(args[0], { acc, x in
                guard let shape = acc.concatenating(with: x) else {
                    throw FunctionTypeError.shapeMismatch(acc, x)
                }
                return shape
            })
        case .binary:
            guard let shape = args[0].broadcasted(to: args[1])
                           ?? args[1].broadcasted(to: args[0]) else {
                throw FunctionTypeError.shapeMismatch(args[0], args[1])
            }
            return shape
        case .unary, .scan, .reduce:
            return args[0]
        case .matrixMultiply:
            guard let shape = args[0].matrixMultiplied(with: args[1]) else {
                throw FunctionTypeError.shapeMismatch(args[0], args[1])
            }
            return shape
        }
    }

}
