//
//  Lexicon.swift
//  DLVM
//
//  Created by Richard Wei on 1/18/17.
//
//

import Foundation


extension ComparisonPredicate : LexicallyConvertible {
    public static let lexicon: [String : ComparisonPredicate] = [
        "lt"  : .lessThan,
        "leq" : .lessThanOrEqualTo,
        "gt"  : .greaterThan,
        "geq" : .greaterThanOrEqualTo,
        "eq"  : .equalTo,
        "neq" : .notEqualTo
    ]
}

extension ArithmeticOperator : LexicallyConvertible {
    public static let lexicon: [String : ArithmeticOperator] = [
        "add"      : .add,
        "sub"      : .subtract,
        "mul"      : .multiply,
        "div"      : .divide,
        "min"      : .min,
        "max"      : .max,
        "truncDiv" : .truncateDivide,
        "floorDiv" : .floorDivide,
        "mod"      : .mod,
        "pow"      : .power
    ]
}

extension ElementwiseFunction : LexicallyConvertible {
    public static let lexicon: [String : ElementwiseFunction] = [
        "sigmoid" : .sigmoid,
        "relu"    : .relu,
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

extension ReductionFunction : LexicallyConvertible {
    public static let lexicon: [String : ReductionFunction] = [
        "reduceAdd"  : .add,  "reduceMul" : .multiply,
        "reduceMin"  : .min,  "reduceMax" : .max,
        "reduceAnd"  : .and,  "reduceOr"  : .or,
        "reduceMean" : .mean
    ]
}

extension BinaryReductionFunction : LexicallyConvertible {
    public static let lexicon: [String : BinaryReductionFunction] = [
        "crossEnt" : .crossEntropy
    ]
}

extension ScanFunction : LexicallyConvertible {
    public static let lexicon: [String : ScanFunction] = [
        "scanAdd" : .add, "scanMul" : .multiply
    ]
}

extension AggregateFunction : LexicallyConvertible {
    public static let lexicon: [String : AggregateFunction] = [
        "softmax"    : .softmax,
        "logSoftmax" : .logSoftmax
    ]
}
