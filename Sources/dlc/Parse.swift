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
fileprivate let number = Lexer.unsignedInteger ^^ { Int($0)! } .. "a number"
fileprivate let lineComments = ("//" ~~> Lexer.string(until: ["\n", "\r"]).maybeEmpty() <~~
                                (newLines | Lexer.end))+
fileprivate let spaces = (Lexer.whitespace | Lexer.tab)+
fileprivate let comma = Lexer.character(",").amid(spaces.?) .. "a comma"
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
     .. "a data type"
}

extension ShapeNode : Parsible {
    static let parser: Parser<ShapeNode> =
        number.nonbacktracking().many(separatedBy: "x")
              .between(Lexer.character("["), Lexer.character("]").! .. "]")
    ^^^ ShapeNode.init
     .. "a shape"
}

extension ImmediateNode : Parsible {
    static let parser: Parser<ImmediateNode> =
        Lexer.signedDecimal ^^ { Double($0)! } ^^^ ImmediateNode.float
      | Lexer.signedInteger ^^ { Int($0)! } ^^^ ImmediateNode.int
      | ( Lexer.token("false") ^^= false
        | Lexer.token("true")  ^^= true  ) ^^^ ImmediateNode.bool
     .. "an immediate value"
}

extension ImmediateValueNode : Parsible {
    static let parser: Parser<ImmediateValueNode> =
        TypeNode.parser <~~ spaces ~~ ImmediateNode.parser.! ^^ ImmediateValueNode.init
     .. "an immediate value"
}

extension VariableNode : Parsible {
    static let parser: Parser<VariableNode> =
        Lexer.character("@") ~~> identifier.! ^^^ VariableNode.global
      | Lexer.character("%") ~~> identifier.! ^^^ VariableNode.temporary
      | ImmediateNode.parser.! ^^^ VariableNode.immediate
     .. "a variable"
}

extension OperandNode : Parsible {
    static let parser: Parser<OperandNode> =
        TypeNode.parser <~~ spaces ^^ curry(OperandNode.init)
     ** (ShapeNode.parser <~~ spaces).?
     ** VariableNode.parser
     .. "an operand"
}

extension InstructionNode : Parsible {

    private static let unaryOpParser: Parser<(OperandNode, SourceRange) -> InstructionNode> =
      ( Lexer.token("sigmoid") ^^= InstructionNode.sigmoid
      | Lexer.token("tanh")    ^^= InstructionNode.tanh
      | Lexer.token("relu")    ^^= InstructionNode.relu
      | Lexer.token("log")     ^^= InstructionNode.log
      | Lexer.token("softmax") ^^= InstructionNode.softmax
      | Lexer.token("neg")     ^^= InstructionNode.neg
      | Lexer.token("load")    ^^= InstructionNode.load
      ) <~~ spaces

    private static let binaryOpParser: Parser<(OperandNode, OperandNode, SourceRange) -> InstructionNode> =
      ( Lexer.token("add")  ^^= InstructionNode.add
      | Lexer.token("sub")  ^^= InstructionNode.sub
      | Lexer.token("mul")  ^^= InstructionNode.mul
      | Lexer.token("div")  ^^= InstructionNode.div
      | Lexer.token("min")  ^^= InstructionNode.min
      | Lexer.token("max")  ^^= InstructionNode.max
      | Lexer.token("tmul") ^^= InstructionNode.tmul
      ) <~~ spaces

    private static let unaryParser: Parser<InstructionNode> =
        unaryOpParser ** OperandNode.parser.!

    private static let binaryParser: Parser<InstructionNode> =
        binaryOpParser ^^ curry
     ** OperandNode.parser.! <~~ comma.! ** OperandNode.parser.!

    private static let concatParser: Parser<InstructionNode> =
        Lexer.token("concat") <~~ spaces ^^= curry(InstructionNode.concat)
     ** OperandNode.parser.nonbacktracking().many(separatedBy: comma)
     ** (Lexer.token("along").amid(spaces) ~~> number).?

    private static let shapeCastParser: Parser<InstructionNode> =
        Lexer.token("shapecast") <~~ spaces ^^= curry(InstructionNode.shapeCast)
     ** OperandNode.parser.! <~~ Lexer.token("to").amid(spaces)
     ** ShapeNode.parser

    private static let typeCastParser: Parser<InstructionNode> =
        Lexer.token("typecast") <~~ spaces ^^= curry(InstructionNode.typeCast)
     ** OperandNode.parser.! <~~ Lexer.token("to").amid(spaces)
     ** TypeNode.parser

    private static let storeParser: Parser<InstructionNode> =
        Lexer.token("store") <~~ spaces ^^= curry(InstructionNode.store)
     ** OperandNode.parser.! <~~ Lexer.token("to").amid(spaces)
     ** OperandNode.parser.!

    static let parser: Parser<InstructionNode> = unaryParser
                                               | binaryParser
                                               | concatParser
                                               | shapeCastParser
                                               | typeCastParser
                                               | storeParser
                                              .. "an instruction"
}

extension InstructionDeclarationNode : Parsible {
    static let parser: Parser<InstructionDeclarationNode> =
        spaces.? ~~> (Lexer.character("%") ~~> identifier <~~ Lexer.character("=").amid(spaces.?)).?
     ^^ curry(InstructionDeclarationNode.init)
     ** InstructionNode.parser
}

extension BasicBlockNode : Parsible {
    static let parser: Parser<BasicBlockNode> =
        identifier ^^ curry(BasicBlockNode.init)
     ** (Lexer.token("gradient").amid(spaces.?).between("(", ")") ^^= true).withDefault(false)
    <~~ Lexer.character(":").amid(spaces.?).! <~~ linebreaks.!
     ** InstructionDeclarationNode.parser.nonbacktracking()
                                  .many(separatedBy: linebreaks).amid(linebreaks.?)
     .. "a basic block"
}

extension DeclarationNode.Role : Parsible {
    static let parser: Parser<DeclarationNode.Role> =
        Lexer.token("input") ^^= .input
      | Lexer.token("parameter") ^^= .parameter
      | Lexer.token("output") ^^= .output
     .. "a global variable role (input, parameter, output)"
}

extension Initializer : Parsible {
    static let parser: Parser<Initializer> =
        ImmediateValueNode.parser ^^^ Initializer.immediate
      | Lexer.token("repeating") ~~> spaces ~~> ImmediateValueNode.parser.! ^^^ Initializer.repeating
      | Lexer.token("random") ~~> Lexer.token("from").amid(spaces) ~~>
        ImmediateValueNode.parser.! ^^ curry(Initializer.random)
        ** (Lexer.token("to").amid(spaces) ~~> ImmediateValueNode.parser.!)
     .. "an initializer"
}

extension DeclarationNode : Parsible {
    static let parser: Parser<DeclarationNode> =
        Lexer.token("declare") ~~> Role.parser.amid(spaces.!) ^^ curry(DeclarationNode.init)
     ** OperandNode.parser.!
     ** (Lexer.character("=").amid(spaces.?) ~~> Initializer.parser.!).?
     .. "a declaration"
}

extension TopLevelItemNode : Parsible {
    static let parser: Parser<TopLevelItemNode> =
        Lexer.token("module") ~~> spaces ~~> identifier.! ^^^ TopLevelItemNode.moduleName
      | DeclarationNode.parser ^^^ TopLevelItemNode.declaration
      | BasicBlockNode.parser ^^^ TopLevelItemNode.basicBlock
}

extension ModuleNode : Parsible {
    static let parser: Parser<ModuleNode> = TopLevelItemNode.parser
                                                            .many(separatedBy: linebreaks)
                                                            .amid(linebreaks.?)
                                        ^^^ ModuleNode.init
}
