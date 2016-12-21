//
//  AST.swift
//  DLVM
//
//  Created by Richard Wei on 12/20/16.
//
//

import enum DLVM.DataType
import struct DLVM.TensorShape
import Parsey

/// Sample
/// ```
/// #type float16
///
/// W1, W2, W3, W4, W5, W6: param[auto] = random(from: 0.0, to: 1.0)
/// b1, b2, b3, b4, b5, b6: param[auto] = 0.0
/// 
/// x: in[2x1]
/// 
/// W1: param[auto] = 0.0
/// h1: layer[4x1] = tanh(W1 x + b1)
/// 
/// recurrent t {
///     h2: layer[16x1] = tanh(W2 [h1.t, h5.(t-1)] + b2)
///     h3: layer[128x1] = tanh(W3 h2 + b3)
///     h4: layer[128x1] = relu(W4 h3 + b4)
///     h5: layer[16x1] = tanh(W5 h4 + b5)
/// }
///
/// h6: layer[16x1] = sigmoid(W6 h5 + b6)
///
/// o: out[16x1] = softmax(W7 h6 + b7)
/// ``````

let identifier = Lexer.regex("[a-zA-Z_][a-zA-Z0-9_]*")
let number = Lexer.signedDecimal ^^ { Int($0)! }
let lineComment = Lexer.regex("//.*?\n")
let space = (Lexer.whitespace | Lexer.tab)+

protocol Parsible {
    static var parser: Parser<Self> { get }
}

enum Macro : Parsible {
    case type(DataType)

    static let parser: Parser<Macro> =
        "#type" ~~> space ~~>
            ( Lexer.token("int8")    ^^= DataType.int8
            | Lexer.token("int16")   ^^= DataType.int16
            | Lexer.token("int32")   ^^= DataType.int32
            | Lexer.token("int64")   ^^= DataType.int64
            | Lexer.token("float8")  ^^= DataType.float8
            | Lexer.token("float16") ^^= DataType.float16
            | Lexer.token("float32") ^^= DataType.float32
            | Lexer.token("float64") ^^= DataType.float64
            ) ^^ Macro.type
}

struct DeclarationType : Parsible {
    enum Role : Parsible {
        case input, output, hidden

        static let parser: Parser<Role> =
            Lexer.token("in")     ^^= .input
          | Lexer.token("out")    ^^= .output
          | Lexer.token("hidden") ^^= .hidden
    }
    
    var role: Role
    var shape: [Int]
    
    static let parser: Parser<DeclarationType> =
        Role.parser ~~
            number.many(separatedBy: Lexer.character("x"))
                  .between("[", "]")
            ^^ { DeclarationType(role: $0, shape: $1) }
}

indirect enum Statement {
    case assignment(String, DeclarationType, Expression)
    case recurrence([Statement])
}

indirect enum Expression : Parsible {
    /// Variable
    case variable(String)
    /// Intrinsic call
    case call(String, [Expression])
    /// Negation
    case negate(Expression)
    /// Element-wise addition
    case add(Expression, Expression)
    /// Element-wise product
    case mul(Expression, Expression)
    /// Tensor product
    case product(Expression, Expression)
    /// Concatenation
    case concat([Expression])

    static var variableParser: Parser<Expression> =
        identifier ^^ Expression.variable

    static var callParser: Parser<Expression> =
        identifier ~~
        termParser
            .many(separatedBy: Lexer.regex(", "))
            .between(Lexer.token("("), Lexer.token(")"))
        ^^ Expression.call

    static var negateParser: Parser<Expression> =
        Lexer.token("-") ~~> termParser
        ^^ Expression.negate

    static var concatParser: Parser<Expression> =
        termParser.many(separatedBy: ", ").between("[", "]")
        ^^ Expression.concat
    
    static var termParser: Parser<Expression> =
        negateParser | variableParser

    static var productParser: Parser<Expression> =
        termParser.infixedLeft(by: Lexer.token(" ")
            ^^= Expression.product)

    static var mulParser: Parser<Expression> =
        productParser.infixedLeft(by: Lexer.token(" * ")
            ^^= Expression.mul)

    static var addParser: Parser<Expression> =
        mulParser.infixedLeft(by: Lexer.token(" + ")
            ^^= Expression.add)

    static var parser: Parser<Expression> = addParser

}
