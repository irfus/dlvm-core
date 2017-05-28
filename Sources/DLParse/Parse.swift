//
//  Lex.swift
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

import func Funky.curry
import Parsey
import CoreTensor
import DLVM

protocol ParseTarget {
    static var parser: Parser<Self> { get }
}

/// Local primitive parsers
private let identifier = Lexer.regex("[a-zA-Z_][a-zA-Z0-9_.]*")
private let unsignedNumber = Lexer.unsignedInteger.flatMap{UInt($0)} .. "a number"
private let number = Lexer.unsignedInteger.flatMap{Int($0)} .. "a number"
private let lineComments = ("//" ~~> Lexer.string(until: ["\n", "\r"]).maybeEmpty() <~~
                            (newLines | Lexer.end))+
private let spaces = (Lexer.whitespace | Lexer.tab)+ .. "a whitespace"
private let comma = Lexer.character(",").amid(spaces.?) .. "a comma"
private let newLines = Lexer.newLine+
private let linebreaks = (newLines | lineComments).amid(spaces.?)+ .. "a linebreak"

infix operator ~> : FunctionCompositionPrecedence
infix operator <~ : FunctionCompositionPrecedence
infix operator ~ : FunctionCompositionPrecedence

extension Parser {
    static func ~><T>(lhs: Parser<Target>, rhs: @autoclosure @escaping () -> Parser<T>) -> Parser<T> {
        return lhs ~~> spaces.? ~~> rhs
    }
    
    static func <~<T>(lhs: Parser<Target>, rhs: @autoclosure @escaping () -> Parser<T>) -> Parser<Target> {
        return lhs <~~ spaces.? <~~ rhs
    }
    
    static func ~<T>(lhs: Parser<Target>, rhs: @autoclosure @escaping () -> Parser<T>) -> Parser<(Target, T)> {
        return lhs <~~ spaces.? ~~ rhs
    }
}

extension FloatingPointSize : ParseTarget {
    static let parser: Parser<FloatingPointSize> =
        Lexer.token("16") ^^= .half
      | Lexer.token("32") ^^= .single
      | Lexer.token("64") ^^= .double
}

extension DataType : ParseTarget {
    static let parser: Parser<DataType> =
        Lexer.token("bool") ^^= .bool
      | Lexer.token("int") ~~> unsignedNumber ^^ DataType.int
      | Lexer.token("float") ~~> FloatingPointSize.parser ^^ DataType.float
}

extension TensorShape : ParseTarget {
    static let parser: Parser<TensorShape> =
        number.nonbacktracking().many(separatedBy: "x")
     ^^ TensorShape.init
     .. "a shape"

}

extension Type : ParseTarget {

    static let tensorParser: Parser<Type> =
        Lexer.character("<") ~> TensorShape.parser.! <~ Lexer.character(".").!
     ~~ DataType.parser.! <~ Lexer.character(">")
     ^^ Type.tensor

    static let arrayParser: Parser<Type> =
        Lexer.character("[") ~> number <~ Lexer.character("x")
     ~~ Type.parser <~ Lexer.character("]")
     ^^ Type.array

    static let parser: Parser<Type> =
        DataType.parser ^^ Type.scalar
      | tensorParser
}

