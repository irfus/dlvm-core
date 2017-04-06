//
//  Intrinsics.swift
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

import DLVM

/// - MARK: TEL intrinsics
let intrinsicTable: [String : OpKind] = [
    /// Elementwise tranfer functions
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

