//
//  Lexicon.swift
//  DLVM
//
//  Created by Richard Wei on 1/18/17.
//
//

import Foundation

public protocol LexicallyConvertible {
    static var lexicon: [String : Self] { get }
}

extension IntegrationOp: LexicallyConvertible {
    public static var lexicon: [String : IntegrationOp] = [
        "softmax"    : .softmax,
        "logSoftmax" : .logSoftmax,
        "argmax"     : .argmax,
        "argmin"     : .argmin
    ]
}

extension ComparisonOp: LexicallyConvertible {
    public static let lexicon: [String : ComparisonOp] = [
        "lt"  : .lessThan,
        "leq" : .lessThanOrEqualTo,
        "gt"  : .greaterThan,
        "geq" : .greaterThanOrEqualTo,
        "eq"  : .equalTo,
        "neq" : .notEqualTo
    ]
}

extension ArithmeticOp: LexicallyConvertible {
    public static let lexicon: [String : ArithmeticOp] = [
        "add"      : .add,
        "sub"      : .subtract,
        "mul"      : .multiply,
        "div"      : .divide,
        "min"      : .min,
        "max"      : .max,
        "truncdiv" : .truncateDivide,
        "floordiv" : .floorDivide,
        "mod"      : .modulo,
        "pow"      : .power,
        "mean"     : .mean
    ]
}

extension ElementwiseOp: LexicallyConvertible {
    public static let lexicon: [String : ElementwiseOp] = [
        "sigmoid" : .sigmoid,
        "tanh"    : .tanh,
        "log"     : .log,
        "exp"     : .exp,
        "neg"     : .neg,
        "sign"    : .sign,
        "square"  : .square,
        "sqrt"    : .sqrt,
        "round"   : .round,
        "rsqrt"   : .rsqrt,
        "ceil"    : .ceil,
        "floor"   : .floor,
        "tan"     : .tan,
        "cos"     : .cos,
        "sin"     : .sin,
        "acos"    : .acos,
        "asin"    : .asin,
        "atan"    : .atan,
        "lgamma"  : .lgamma,
        "digamma" : .digamma,
        "erf"     : .erf,
        "erfc"    : .erfc,
        "rint"    : .rint
    ]
}

extension BooleanOp: LexicallyConvertible {
    public static var lexicon: [String : BooleanOp] = [
        "and" : .and,
        "or"  : .or,
        "xor" : .xor
    ]
}
