//
//  Types.swift
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

import CoreTensor
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

    func makeInstruction(with arguments: [Use]) throws -> InstructionKind {
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

///
/// ## Experimental: type system
/// 

public typealias DimensionVariable = String

public enum MetaDimension {
    case concrete(Int)
    case one(DimensionVariable)
    case many(DimensionVariable)
    case manyOrNone(DimensionVariable)
    case optional(DimensionVariable)
    case transpose(DimensionVariable)
}

public struct MetaShape {
    let dimensions: [MetaDimension]
}

public struct FunctionType {
    public let generics: Set<DimensionVariable>
    public let arguments: [MetaShape]
    public let result: MetaShape
}

extension MetaDimension : Equatable {
    public static func == (lhs: MetaDimension, rhs: MetaDimension) -> Bool {
        switch (lhs, rhs) {
        case let (.concrete(i1), .concrete(i2)):
            return i1 == i2
        case let (.one(d1), .one(d2)),
             let (.many(d1), .many(d2)),
             let (.manyOrNone(d1), .manyOrNone(d2)),
             let (.optional(d1), .optional(d2)),
             let (.transpose(d1), .transpose(d2)):
            return d1 == d2
        default:
            return false
        }
    }
}

extension MetaShape : Equatable {
    public static func == (lhs: MetaShape, rhs: MetaShape) -> Bool {
        return lhs.dimensions == rhs.dimensions
    }
}

extension FunctionType : Equatable {
    public static func == (lhs: FunctionType, rhs: FunctionType) -> Bool {
        /// Alpha equivalence
        /// 1. If they have different generic count or argument count, fail bigly
        guard lhs.generics.count == rhs.generics.count,
              lhs.arguments.count == rhs.arguments.count
            else { return false }
        /// TODO
        fatalError("TODO")
    }
}

public extension MetaDimension {
    func contains(_ variable: DimensionVariable) -> Bool {
        switch self {
        case .one(variable),
             .many(variable),
             .manyOrNone(variable),
             .optional(variable),
             .transpose(variable):
            return true
        default:
            return false
        }
    }
}

public extension MetaShape {
    func contains(_ variable: DimensionVariable) -> Bool {
        return dimensions.contains(where: {$0.contains(variable)})
    }
}

public extension FunctionType {
    var isValid: Bool {
        /// TODO: Invalid cases
        /// 1. Generics exist but not occur in arguments
        /// 2. Undeclared generic variable
        /// 3. Nondeterministic position of dimension
        fatalError("TODO")
    }
}
