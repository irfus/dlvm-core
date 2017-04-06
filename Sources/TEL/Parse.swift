//
//  Parse.swift
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

import Parsey
import func Funky.curry
import func Funky.flip

/// Local primitive parsers
fileprivate let identifier = Lexer.regex("[a-zA-Z_][a-zA-Z0-9_]*")
fileprivate let number = Lexer.unsignedInteger.flatMap{Int($0)} .. "a number"
fileprivate let lineComments = ("//" ~~> Lexer.string(until: ["\n", "\r"]).maybeEmpty() <~~
                                (newLines | Lexer.end))+
fileprivate let spaces = (Lexer.whitespace | Lexer.tab)+
fileprivate let comma = Lexer.character(",").amid(spaces.?)
fileprivate let newLines = Lexer.newLine+
fileprivate let linebreaks = (newLines | lineComments).amid(spaces.?)+ .. "a linebreak"

fileprivate let keywords: Set<String> = [
    "as", "in", "out", "param", "hidden", "recurrent", "random"
]

public protocol Parsible {
    static var parser: Parser<Self> { get }
}

extension Variable : Parsible {
    public static let parser =
        /*
        identifier ^^ curry(Variable.recurrent)
     ** ("[" ~~> identifier.!)
     ** ("-" ~~> number ^^ {-$0}).withDefault(0) <~~ "]"
        */
        identifier ^^^ Variable.simple
}

extension Attribute: Parsible {

    public static let typeParser: Parser<Attribute> =
        "type" ~~> (spaces ~~> identifier.! .. "a data type")
     ^^^ Attribute.type

    public static let nameParser: Parser<Attribute> =
        "module" ~~> (spaces ~~> identifier.! .. "a name")
     ^^^ Attribute.name

    public static let parser: Parser<Attribute> =
        typeParser | nameParser
}

extension Role : Parsible {
    public static let parser: Parser<Role> =
        Lexer.token("in")     ^^= .input
      | Lexer.token("out")    ^^= .output
      | Lexer.token("hidden") ^^= .hidden
      | Lexer.token("param")  ^^= .parameter
     .. "a role: in, out, hidden or param"
}

extension Declaration : Parsible {

    private static let assignmentParser: Parser<Declaration> =
        Variable.parser
     ^^ curry(Declaration.assignment)
     ** (Lexer.character(":").amid(spaces.?) ~~> Role.parser.!)
     ** number.nonbacktracking()
              .many(separatedBy: Lexer.character("x"))
              .between(Lexer.character("[").!, Lexer.character("]").! .. "]")
           .. "a shape, e.g. [2x4], [1x2x3]"
     ** (Lexer.character("=").amid(spaces.?) ~~> Expression.parser.!).?

    private static let recurrenceParser: Parser<Declaration> =
        Lexer.token("recurrent") ~~>
        identifier.nonbacktracking().between(spaces, spaces | linebreaks)
     .. "a time step variable"
     ^^ curry(Declaration.recurrence)
     ** parser.many(separatedBy: linebreaks)
              .between(Lexer.character("{").! .. "{" ~~> linebreaks.!,
                       linebreaks.! ~~> Lexer.character("}").! .. "}")

    public static let parser = assignmentParser
                             | recurrenceParser
                            .. "a declaration"
}

extension Constant : Parsible {
    public static let parser: Parser<Constant> =
        Lexer.signedDecimal.flatMap{Double($0)} ^^^ { .float($0, $1) }
      | Lexer.signedInteger.flatMap{Int($0)} ^^^ { .int($0, $1) }
}

// MARK: - Parser
extension Expression : Parsible {

    ///
    /// Non-left-recursive grammar begin
    ///

    private static let constantParser: Parser<Expression> =
        Constant.parser ^^^ Expression.constant

    private static let variableParser: Parser<Expression> =
        Variable.parser ^^^ Expression.variable

    private static let randomParser: Parser<Expression> =
        Lexer.token("random") ~~> Lexer.character("(").amid(spaces.?)
     ^^= curry(Expression.random)
     ** Constant.parser.! <~~ comma.!
     ** Constant.parser.! <~~ (Lexer.character(")").!).amid(spaces.?)

    private static let callParser: Parser<Expression> =
        identifier ^^ curry(Expression.call)
     ** parser.nonbacktracking()
              .many(separatedBy: comma)
              .amid(spaces.?)
              .between(Lexer.character("("), Lexer.character(")").!)

    private static let negateParser: Parser<Expression> =
        "-" ~~> parser ^^^ Expression.negate

    private static let concatParser: Parser<Expression> =
        parser.nonbacktracking()
              .many(separatedBy: comma)
              .between("[", "]")
     ^^ curry(Expression.concat)
     ** ("@" ~~> number.!).withDefault(0)

    private static let parenthesizedParser: Parser<Expression> =
        "(" ~~> parser.amid(spaces.?) <~~ ")"

    /// Composite parser for a term of an infix expression
    private static let termParser = randomParser
                                  | callParser
                                  | parenthesizedParser
                                  | negateParser
                                  | constantParser
                                  | concatParser
                                  | variableParser
                                 .. "an expression"

    ///
    /// Operators begin
    ///

    private static let transposeOperatorParser: Parser<(Expression, SourceRange) -> Expression> =
        Lexer.token("^T") ^^= Expression.transpose

    private static let transposeParser: Parser<Expression> =
        termParser.suffixed(by: transposeOperatorParser)

    private static let reshapeOperatorParser: Parser<(Expression, SourceRange) -> Expression> =
        spaces ~~> Lexer.token("as") ~~>
        (spaces.! .. "a space followed by a shape") ~~>
        number.nonbacktracking()
              .many(separatedBy: Lexer.character("x"))
              .between(Lexer.character("[").!, Lexer.character("]").! .. "]")
           .. "a shape, e.g. [2x4], [1x2x3]"
     ^^ { target in { Expression.reshape($0, shape: target, $1) } }


    private static let reshapeParser: Parser<Expression> =
        transposeParser.suffixed(by: reshapeOperatorParser)
    
    /// Tensor product: W . x | W • x
    /// - Priority: high
    private static let productParser: Parser<Expression> =
        reshapeParser.infixedLeft(by:
            Lexer.anyCharacter(in: [".", "•", "⊗"]).amid(spaces.?)
            ^^= Expression.product)

    /// Tensor element-wise multiplication: x * y
    /// - Priority: medium
    private static let mulParser: Parser<Expression> =
        productParser.infixedLeft(by:
            ( Lexer.character("*") ^^= { Expression.infixOp(.mul, $0, $1, $2) }
            | Lexer.character("/") ^^= { Expression.infixOp(.div, $0, $1, $2) } )
        .amid(spaces.?))

    /// Tensor element-wise addition/subtraction: x + b, x - b
    /// - Priority: low
    private static let addParser: Parser<Expression> =
        mulParser.infixedLeft(by:
            ( Lexer.character("+") ^^= { Expression.infixOp(.add, $0, $1, $2) }
            | Lexer.character("-") ^^= { Expression.infixOp(.sub, $0, $1, $2) } )
        .amid(spaces.?))

    /// Parser head - add operator
    public static let parser: Parser<Expression> =
        addParser .. "an expression"

}

extension Statement : Parsible {
    public static let parser: Parser<Statement> =
        Attribute.parser   ^^^ Statement.attribute
      | Declaration.parser ^^^ Statement.declaration
     .. "a statement"
}

extension ProgramTree : Parsible {
    public static let parser: Parser<ProgramTree> =
        Statement.parser.manyOrNone(separatedBy: linebreaks)
                        .amid(linebreaks.?)
     ^^^ ProgramTree.init
}
