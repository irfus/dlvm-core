//
//  Parse.swift
//  DLVM
//
//  Created by Richard Wei on 12/22/16.
//
//

import Parsey
import func Funky.curry
import func Funky.flip

/// Local primitive parsers
fileprivate let identifier = Lexer.regex("[a-zA-Z_][a-zA-Z0-9_.]*")
fileprivate let number = Lexer.unsignedInteger ^^ { Int($0)! }
fileprivate let lineComments = ("//" ~~> Lexer.string(until: "\n") <~~ Lexer.newLine)+
fileprivate let spaces = (Lexer.whitespace | Lexer.tab)+
fileprivate let comma = Lexer.character(",").amid(spaces.?)
fileprivate let newLines = Lexer.newLine+
fileprivate let linebreaks = (newLines | lineComments).amid(spaces.?)+ .. "a linebreak"

protocol Parsible {
    static var parser: Parser<Self> { get }
}

extension TypeNode : Parsible {
    static let parser: Parser<TypeNode> =
        Lexer.character("f") ~~> number ^^^ TypeNode.float
      | Lexer.character("i") ~~> number ^^^ TypeNode.int
      | Lexer.character("b") ~~> number ^^^ TypeNode.bool
}

extension ShapeNode : Parsible {
    static let parser: Parser<ShapeNode> =
        number.many(separatedBy: comma)
              .between(Lexer.character("["), Lexer.character("]").!)
          ^^^ ShapeNode.init
}

extension ImmediateNode : Parsible {
    static let parser: Parser<ImmediateNode> =
        Lexer.signedDecimal ^^ { Double($0)! } ^^^ ImmediateNode.float
      | Lexer.signedInteger ^^ { Int($0)! } ^^^ ImmediateNode.int
      | ( Lexer.token("false") ^^= false
        | Lexer.token("true")  ^^= true  ) ^^^ ImmediateNode.bool
}

extension VariableNode : Parsible {
    static let parser: Parser<VariableNode> =
        Lexer.character("@") ~~> identifier ^^^ VariableNode.global
      | Lexer.character("%") ~~> identifier ^^^ VariableNode.temporary
      | ImmediateNode.parser ^^^ VariableNode.immediate
}

extension OperandNode : Parsible {
    static let parser: Parser<OperandNode> =
        TypeNode.parser ^^ curry(OperandNode.init)
     ** ShapeNode.parser.?
     ** VariableNode.parser
}

extension InstructionNode : Parsible {

    private static let unaryOpParser: Parser<(OperandNode, SourceRange) -> InstructionNode> =
      ( Lexer.token("sigmoid") ^^= InstructionNode.sigmoid
      | Lexer.token("tanh") ^^= InstructionNode.tanh
      | Lexer.token("relu") ^^= InstructionNode.relu
      | Lexer.token("log") ^^= InstructionNode.log
      | Lexer.token("softmax") ^^= InstructionNode.softmax
      | Lexer.token("neg") ^^= InstructionNode.neg
      ) <~~ spaces

    private static let binaryOpParser: Parser<(OperandNode, OperandNode, SourceRange) -> InstructionNode> =
      ( Lexer.token("add") ^^= InstructionNode.add
      | Lexer.token("sub") ^^= InstructionNode.sub
      | Lexer.token("mul") ^^= InstructionNode.mul
      | Lexer.token("div") ^^= InstructionNode.div
      | Lexer.token("min") ^^= InstructionNode.min
      | Lexer.token("max") ^^= InstructionNode.max
      | Lexer.token("tmul") ^^= InstructionNode.tmul
      ) <~~ spaces

    private static let unaryParser: Parser<InstructionNode> =
        unaryOpParser ** OperandNode.parser

    private static let binaryParser: Parser<InstructionNode> =
        binaryOpParser ^^ curry
     ** OperandNode.parser <~~ comma ** OperandNode.parser

    private static let concatParser: Parser<InstructionNode> =
        Lexer.token("concat") <~~ spaces ^^= curry(InstructionNode.concat)
     ** OperandNode.parser.many(separatedBy: comma)
     ** (Lexer.token("along").amid(spaces) ~~> number).?

    private static let shapeCastParser: Parser<InstructionNode> =
        Lexer.token("shapecast") <~~ spaces ^^= curry(InstructionNode.shapeCast)
     ** OperandNode.parser <~~ Lexer.token("to").amid(spaces)
     ** ShapeNode.parser

    private static let typeCastParser: Parser<InstructionNode> =
        Lexer.token("typecast") <~~ spaces ^^= curry(InstructionNode.typeCast)
     ** OperandNode.parser <~~ Lexer.token("to").amid(spaces)
     ** TypeNode.parser

    private static let storeParser: Parser<InstructionNode> =
        Lexer.token("store") <~~ spaces ^^= curry(InstructionNode.store)
     ** OperandNode.parser <~~ Lexer.token("to").amid(spaces)
     ** OperandNode.parser

    static let parser: Parser<InstructionNode> = unaryParser
                                               | binaryParser
                                               | concatParser
                                               | shapeCastParser
                                               | typeCastParser
                                               | storeParser
}

extension InstructionDeclarationNode : Parsible {
    static let parser: Parser<InstructionDeclarationNode> =
        identifier <~~ Lexer.character("=").amid(spaces.?)
     ^^ curry(InstructionDeclarationNode.init)
     ** InstructionNode.parser
}

extension BasicBlockNode : Parsible {
    static let parser: Parser<BasicBlockNode> =
        identifier ^^ curry(BasicBlockNode.init)
     ** (Lexer.token("gradient").amid(spaces.?).between("(", ")") ^^= true)
        .withDefault(false)
    <~~ spaces.? <~~ Lexer.character(":").amid(spaces.?) <~~ linebreaks
     ** InstructionNode.parser.many(separatedBy: linebreaks)
                              .amid(linebreaks.?)
}
