//
//  Intrinsics.swift
//  DLVM
//
//  Created by Richard Wei on 1/29/17.
//
//

import DLVM

/// - MARK: TEL intrinsics
let intrinsicTable: [String : OpKind] = [
    /// Elementwise tranfer functions
    "sigmoid" : .unary(.elementwise(.sigmoid)),
    "log" : .unary(.elementwise(.log)),
    "exp" : .unary(.elementwise(.exp)),
    "tan" : .unary(.elementwise(.tan)),
    "tanh" : .unary(.elementwise(.tanh)),
    "atan" : .unary(.elementwise(.atan)),
    "sin" : .unary(.elementwise(.sin)),
    "asin" : .unary(.elementwise(.asin)),
    "cos" : .unary(.elementwise(.cos)),
    "acos" : .unary(.elementwise(.acos)),
    "sqrt" : .unary(.elementwise(.sqrt)),
    "ceil" : .unary(.elementwise(.ceil)),
    "floor" : .unary(.elementwise(.floor)),

    /// Integration functions
    "softmax" : .unary(.integration(.softmax)),
    "logSoftmax" : .unary(.integration(.logSoftmax)),

    /// Logical operators (not yet supported)
//    "&&" : .binary(.associative(.boolean(.and))),
//    "||" : .binary(.associative(.boolean(.or))),
//    "^^" : .binary(.associative(.boolean(.xor))),

    /// Arithmetic operators
//    "+" : .binary(.associative(.arithmetic(.add))),
//    "-" : .binary(.associative(.arithmetic(.subtract))),
//    "*" : .binary(.associative(.arithmetic(.multiply))),
//    "/" : .binary(.associative(.arithmetic(.divide))),
    "min" : .binary(.associative(.arithmetic(.min))),
    "max" : .binary(.associative(.arithmetic(.max))),
    "pow" : .binary(.associative(.arithmetic(.power))),
    "mod" : .binary(.associative(.arithmetic(.modulo))),

    /// Comparison operators (not yet supported)
//    ">" : .binary(.comparison(.greaterThan)),
//    "<" : .binary(.comparison(.lessThan)),
//    ">=" : .binary(.comparison(.greaterThanOrEqualTo)),
//    "<=" : .binary(.comparison(.lessThanOrEqualTo)),
//    "==" : .binary(.comparison(.equalTo)),
//    "!=" : .binary(.comparison(.notEqualTo)),
]

