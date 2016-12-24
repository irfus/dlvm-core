//
//  AST.swift
//  DLVM
//
//  Created by Richard Wei on 12/20/16.
//
//

import enum DLVM.DataType
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
fileprivate let lineComments = ("//" ~~> Lexer.string(until: "\n") <~~ Lexer.newLine)+
fileprivate let spaces = (Lexer.whitespace | Lexer.tab)+
fileprivate let newLines = Lexer.newLine+
fileprivate let linebreaks = (newLines | lineComments).amid(spaces.?)+ .. "a linebreak"

public protocol Parsible {
    static var parser: Parser<Self> { get }
}

///
/// AST begin
///

public enum Macro {
    case type(DataType)
}

public enum Variable {
    case simple(String)
    case recurrent(String, timestep: String, offset: Int)

    var name: String {
        switch self {
        case let .simple(name), let .recurrent(name, _, _):
            return name
        }
    }
}

public enum Role {
    case input, output, hidden, parameter
}

public indirect enum Expression {
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

public indirect enum Declaration {
    case assignment(Variable, Role, [Int], Expression?)
    case recurrence(String, [Declaration])
}

public enum Statement {
    case macro(Macro)
    case declaration(Declaration)
}

public struct ProgramTree {
    public let statements: [Statement]
}

///
/// Parsers begin
///

extension Variable : Parsible {
    public static let parser =
        identifier ^^ curry(Variable.recurrent)
     ** ("[" ~~> identifier.!)
     ** ("-" ~~> number ^^ {-$0}).withDefault(0) <~~ "]"
      | identifier ^^ Variable.simple
}

extension Macro : Parsible {
    public static let parser: Parser<Macro> =
        "#type" ~~> spaces ~~>
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

extension Role : Parsible {
    public static let parser: Parser<Role> =
        Lexer.token("in")     ^^= .input
      | Lexer.token("out")    ^^= .output
      | Lexer.token("hidden") ^^= .hidden
      | Lexer.token("param")  ^^= .parameter
}

extension Declaration : Parsible {

    private static let assignmentParser: Parser<Declaration> =
        Variable.parser
     ^^ curry(Declaration.assignment)
     ** (Lexer.character(":").amid(spaces.?) !~~> Role.parser)
     ** number.nonbacktracking()
              .many(separatedBy: Lexer.character("x"))
              .between(Lexer.character("[").!, Lexer.character("]").!)
     ** (Lexer.character("=").amid(spaces.?) ~~> Expression.parser.!).?

    private static let recurrenceParser: Parser<Declaration> =
        Lexer.token("recurrent") ~~>
        identifier.nonbacktracking().amid(spaces)
     ^^ curry(Declaration.recurrence)
     ** parser.many(separatedBy: linebreaks)
              .between(Lexer.character("{").! ~~> linebreaks.!,
                       linebreaks.! ~~> Lexer.character("}").!)

    public static let parser = assignmentParser
                             | recurrenceParser
                            .. "a declaration"
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
        parser.nonbacktracking()
              .many(separatedBy: Lexer.character(",").amid(spaces.?))
              .between(Lexer.token("("), Lexer.token(")").!)
     ^^ Expression.call

    private static let negateParser: Parser<Expression> =
        "-" ~~> parser.! ^^ Expression.negate

    private static let concatParser: Parser<Expression> =
        parser.nonbacktracking()
              .many(separatedBy: Lexer.character(",").amid(spaces.?))
              .between("[", "]")
     ^^ Expression.concat

    private static let parenthesizedParser: Parser<Expression> =
        "(" ~~> parser.amid(spaces.?) <~~ ")"

    /// Composite parser for a term of an infix expression
    private static let termParser = callParser
                                  | parenthesizedParser
                                  | negateParser
                                  | floatParser
                                  | intParser
                                  | concatParser
                                  | variableParser
                                 .. "an expression"

    ///
    /// Infix operators begin
    ///
    
    /// Tensor product: W x
    /// - Priority: high
    private static let productParser: Parser<Expression> =
        termParser.infixedLeft(by: spaces ^^= Expression.product)

    /// Tensor element-wise multiplication: x * y
    /// - Priority: medium
    private static let mulParser: Parser<Expression> =
        productParser.infixedLeft(by: Lexer.character("*").amid(spaces.?)
            ^^= Expression.mul)

    /// Tensor element-wise addition: x + b
    /// - Priority: low
    private static let addParser: Parser<Expression> =
        mulParser.infixedLeft(by: Lexer.character("+").amid(spaces.?)
            ^^= Expression.add)

    /// Parser head - add operator
    public static let parser: Parser<Expression> =
        addParser .. "an expression"

}

extension Statement : Parsible {
    public static let parser: Parser<Statement> =
        Macro.parser       ^^ Statement.macro
      | Declaration.parser ^^ Statement.declaration
     .. "a statement"
}

extension ProgramTree : Parsible {
    public static let parser: Parser<ProgramTree> =
        Statement.parser.manyOrNone(separatedBy: linebreaks)
                        .amid(linebreaks.?)
     ^^ ProgramTree.init
}
