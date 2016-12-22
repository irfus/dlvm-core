//
//  AST.swift
//  DLVM
//
//  Created by Richard Wei on 12/20/16.
//
//

import enum DLVM.DataType
import struct DLVM.TensorShape
import func Funky.curry
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

/// Local primitive parsers
fileprivate let identifier = Lexer.regex("[a-zA-Z_][a-zA-Z0-9_]*")
fileprivate let number = Lexer.unsignedInteger ^^ { Int($0)! }
fileprivate let lineComment = Lexer.regex("//.*?") <~~ Lexer.newLine
fileprivate let space = (Lexer.whitespace | Lexer.tab)+

protocol Parsible {
    static var parser: Parser<Self> { get }
}

///
/// AST begin
///

enum Macro {
    case type(DataType)
}

enum Variable {
    case simple(String)
    case recurrent(String, timestep: String, offset: Int)
}

struct DeclarationType {
    enum Role {
        case input, output, hidden, parameter
    }
    var role: Role
    var shape: [Int]
}

indirect enum Expression {
    /// Integer
    case int(Int)
    /// Float
    case float(Float)
    /// Variable
    case variable(Variable)
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
}

indirect enum Declaration {
    case assignment(Variable, DeclarationType, Expression)
    case recurrence(String, [Declaration])
}

struct Program {
    let macros: [Macro]
    let declarations: [Declaration]
}

///
/// Parsers begin
///

extension Variable : Parsible {
    static let parser =
        identifier ^^ Variable.simple
      | identifier ^^ curry(Variable.recurrent)
     ** ("[" ~~> identifier)
     ** (number ^^ {-$0} <~~ "]")
}

extension Macro : Parsible {
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

extension DeclarationType.Role : Parsible {
    static let parser: Parser<DeclarationType.Role> =
        Lexer.token("in")     ^^= .input
      | Lexer.token("out")    ^^= .output
      | Lexer.token("hidden") ^^= .hidden
      | Lexer.token("param")  ^^= .parameter
}

extension DeclarationType : Parsible {
    static let parser: Parser<DeclarationType> =
        Role.parser ~~
        number.many(separatedBy: Lexer.character("x")).!
              .between(Lexer.character("[").!, Lexer.character("]").!)
     ^^ { DeclarationType(role: $0, shape: $1) }
}

extension Declaration : Parsible {
    
    private static let assignmentParser =
        Variable.parser
     ^^ curry(Declaration.assignment)
     ** (Lexer.character(":").amid(space.?) ~~> DeclarationType.parser.!)
     ** (Lexer.character("=").amid(space.?) ~~> Expression.parser.!)

    private static let recurrenceParser =
        Lexer.token("recurrent") ~~> identifier.amid(space)
     ^^ curry(Declaration.recurrence)
     ** parser.many(separatedBy: Lexer.newLines)
              .between(Lexer.character("{") ~~> Lexer.newLines,
                       Lexer.newLines ~~> Lexer.character("}"))

    static let parser: Parser<Declaration> = (
        assignmentParser
      | recurrenceParser
    ).amid(space.?)
}

// MARK: - Parser
extension Expression : Parsible {

    ///
    /// Non-left-recursive grammar begin
    ///

    private static let intParser: Parser<Expression> =
        Lexer.signedInteger ^^ { .int(Int($0)!) }

    private static let floatParser: Parser<Expression> =
        Lexer.signedDecimal ^^ { .float(Float($0)!) }

    private static let variableParser: Parser<Expression> =
        Variable.parser ^^ Expression.variable

    private static let callParser: Parser<Expression> =
        identifier ~~
        parser.many(separatedBy: Lexer.regex(", "))
              .between(Lexer.token("("), Lexer.token(")"))
     ^^ Expression.call

    private static let negateParser: Parser<Expression> =
        "-" ~~> termParser ^^ Expression.negate

    private static let concatParser: Parser<Expression> =
        parser.many(separatedBy: Lexer.character(",").amid(space.?))
              .between("[", "]")
     ^^ Expression.concat

    private static let parenthesizedParser: Parser<Expression> =
        "(" ~~> parser.amid(space.?) <~~ ")"

    /// Composite parser for a term of an infix expression
    private static let termParser = callParser
                                  | negateParser
                                  | concatParser
                                  | variableParser
                                  | parenthesizedParser

    ///
    /// Infix operators begin
    ///
    
    /// Tensor product: W x
    /// - Priority: high
    private static let productParser: Parser<Expression> =
        termParser.infixedLeft(by: space ^^= Expression.product)

    /// Tensor element-wise multiplication: x * y
    /// - Priority: medium
    private static let mulParser: Parser<Expression> =
        productParser.infixedLeft(by: Lexer.character("*").amid(space.?)
            ^^= Expression.mul)

    /// Tensor element-wise addition: x + b
    /// - Priority: low
    private static let addParser: Parser<Expression> =
        mulParser.infixedLeft(by: Lexer.character("+").amid(space.?)
            ^^= Expression.add)

    /// Parser head - add operator
    static let parser: Parser<Expression> = addParser.amid(space.?)

}
