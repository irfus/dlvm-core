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
        "lessThan"           : .lessThan,
        "lessThanOrEqual"    : .lessThanOrEqualTo,
        "greaterThan"        : .greaterThan,
        "greaterThanOrEqual" : .greaterThanOrEqualTo,
        "equal"              : .equalTo,
        "notEqual"           : .notEqualTo
    ]
}

extension ArithmeticOp: LexicallyConvertible {
    public static let lexicon: [String : ArithmeticOp] = [
        "add"            : .add,
        "subtract"       : .subtract,
        "multiply"       : .multiply,
        "divide"         : .divide,
        "min"            : .min,
        "max"            : .max,
        "truncateDivide" : .truncateDivide,
        "floorDivide"    : .floorDivide,
        "modulo"         : .modulo,
        "power"          : .power,
        "mean"           : .mean
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
