//
//  Op.swift
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

public enum ComparisonOp {
    case lessThan, lessThanOrEqual
    case greaterThan, greaterThanOrEqual
    case equal, notEqual
}

public enum BooleanOp {
    case and, or, xor
}

public enum ArithmeticOp {
    case add, subtract, multiply, divide, min, max
    case truncateDivide, floorDivide, modulo, power, mean
}

public enum ElementwiseOp {
    case tanh, log, exp, neg, sign, square, sqrt
    case round, rsqrt, ceil, floor
    case tan, cos, sin, acos, asin, atan
    case lgamma, digamma, erf, erfc, rint
}

public enum UnaryOp {
    case elementwise(ElementwiseOp)
}

public enum AssociativeOp {
    case boolean(BooleanOp)
    case arithmetic(ArithmeticOp)
}

public enum BinaryOp {
    case associative(AssociativeOp)
    case comparison(ComparisonOp)
}

/// - TODO: Add custom op
public enum OpKind {
    case unary(UnaryOp)        /// Monomorphic
    case binary(BinaryOp)      /// Monomorphic
    case scan(AssociativeOp)   /// Scan
    case reduce(AssociativeOp) /// Reduce
    case matrixMultiply        /// Matrix multiplication
    case concatenate           /// Concatenation
//    case aggregate(AggregateOp)
}

public extension OpKind {

    var argumentCount: Int {
        switch self {
        case .unary: return 1
        case .binary: return 2
        case .matrixMultiply: return 2
        case .reduce: return 1
        case .scan: return 1
        case .concatenate: return Int.max
//        case .aggregate(let aggr): return aggr.argumentCount
        }
    }

    func resultShape(forArguments args: [TensorShape]) -> TensorShape? {
        guard !args.isEmpty else { return nil }
        switch self {
        case .concatenate:
            return args.dropFirst().reduce(args[0], { acc, x in acc?.concatenating(with: x) })
        case .unary where args.count == 1:
            return args[0]
        case .binary where args.count == 2:
            return args[0].mutuallyBroadcasted(with: args[1])
        case .scan, .reduce:
            return args[0]
        case .matrixMultiply:
            return args[0].matrixMultiplied(with: args[1])
//        case .aggregate(let aggr):
//            return aggr.resultShape(forArguments: args)
        default:
            return nil
        }
    }

}

/* A prototype of the implementation of custom ops

public protocol AggregateOp {
    var argumentCount: Int { get }
    func resultShape(forArguments args: [TensorShape]) -> TensorShape?
}

public struct Convolution2D : AggregateOp {
    public enum PaddingAlgorithm {
        case same, valid
    }

    public var argumentCount: Int {
        return 4
    }

    public func resultShape(forArguments args: [TensorShape]) -> TensorShape? {
        /// TODO
        DLUnimplemented()
    }
}
 
 */
