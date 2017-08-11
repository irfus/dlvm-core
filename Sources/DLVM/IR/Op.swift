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

import CoreTensor

// MARK: - Operator definition

public enum ComparisonOp {
    case lessThan, lessThanOrEqual
    case greaterThan, greaterThanOrEqual
    case equal, notEqual
}

public enum UnaryOp {
    case sinh, cosh, tanh, log, exp, negate, sign, square, sqrt
    case round, rsqrt, ceil, floor
    case tan, cos, sin, acos, asin, atan
    case lgamma, digamma, erf, erfc, rint
    case not
}

public enum AssociativeOp {
    case and, or
    case add, subtract, multiply, divide, min, max
    case truncateDivide, floorDivide, modulo, power, mean
}

public enum BinaryOp {
    case associative(AssociativeOp)
    case comparison(ComparisonOp)
}

public enum ReductionCombinator {
    case function(Use)
    case op(AssociativeOp)
}

// MARK: - Operator kinds and properties

public enum OpKind {
    case map(UnaryOp)          /// Unary elementwise
    case zipWith(BinaryOp)     /// Binary elementwise
    case scan(AssociativeOp)   /// Scan
    case reduce(AssociativeOp) /// Reduce
    case dot                   /// vector dot, matrix-vector mult & matrix-matrix mult
    case concatenate           /// Concatenation
}

public extension AssociativeOp {
    var isBoolean: Bool {
        switch self {
        case .and, .or: return true
        default: return false
        }
    }
}

extension BinaryOp : Equatable {
    public static func == (lhs: BinaryOp, rhs: BinaryOp) -> Bool {
        switch (lhs, rhs) {
        case let (.associative(o1), .associative(o2)): return o1 == o2
        case let (.comparison(o1), .comparison(o2)): return o1 == o2
        default: return false
        }
    }
}

public extension OpKind {
    var argumentCount: Int {
        switch self {
        case .map: return 1
        case .zipWith: return 2
        case .dot: return 2
        case .reduce: return 1
        case .scan: return 1
        case .concatenate: return Int.max
        }
    }

    func resultShape(forArguments args: [TensorShape]) -> TensorShape? {
        guard !args.isEmpty else { return nil }
        switch self {
        case .concatenate:
            return args.dropFirst().reduce(args[0], { acc, x in acc?.concatenating(with: x) })
        case .map where args.count == 1:
            return args[0]
        case .zipWith(_) where args.count == 2:
            return args[0].broadcast(with: args[1])
        case .scan, .reduce:
            return args[0]
        case .dot:
            return args[0].matrixMultiplied(by: args[1])
        default:
            return nil
        }
    }
}
