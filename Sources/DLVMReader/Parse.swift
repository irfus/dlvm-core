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

@inline(__always)
fileprivate func curry<A, B, C, D, E, F, G>(_ f: @escaping (A, B, C, D, E, F) -> G)
        -> (A) -> (B) -> (C) -> (D) -> (E) -> (F) -> G {
    return { w in { x in { y in { z in { a in { b in f(w, x, y, z, a, b) } } } } } }
}

/// Local primitive parsers
fileprivate let identifier = Lexer.regex("[a-zA-Z_][a-zA-Z0-9_.]*")
fileprivate let number = Lexer.unsignedInteger.flatMap{Int($0)} .. "a number"
fileprivate let lineComments = ("//" ~~> Lexer.string(until: ["\n", "\r"]).maybeEmpty() <~~
                                (newLines | Lexer.end))+
fileprivate let spaces = (Lexer.whitespace | Lexer.tab)+ .. "a whitespace"
fileprivate let comma = Lexer.character(",").amid(spaces.?) .. "a comma"
fileprivate let newLines = Lexer.newLine+
fileprivate let linebreaks = (newLines | lineComments).amid(spaces.?)+ .. "a linebreak"

public protocol Parsible {
    static var parser: Parser<Self> { get }
}

extension TypeNode : Parsible {
    public static let parser: Parser<TypeNode> =
        Lexer.character("f") ~~> number ^^^ TypeNode.float
      | Lexer.character("i") ~~> number ^^^ TypeNode.int
      | Lexer.character("b") ~~> number ^^^ TypeNode.bool
     .. "a data type"
}

extension ShapeNode : Parsible {
    public static let parser: Parser<ShapeNode> =
        number.nonbacktracking().many(separatedBy: "x")
              .between(Lexer.character("["), Lexer.character("]").! .. "]")
    ^^^ ShapeNode.init
     .. "a shape"
}

extension ImmediateNode : Parsible {
    public static let parser: Parser<ImmediateNode> =
        Lexer.signedDecimal.flatMap{Double($0)} ^^^ ImmediateNode.float
      | Lexer.signedInteger.flatMap{Int($0)} ^^^ ImmediateNode.int
      | ( Lexer.token("false") ^^= false
        | Lexer.token("true")  ^^= true  ) ^^^ ImmediateNode.bool
     .. "an immediate value"
}

extension ImmediateValueNode : Parsible {
    public static let parser: Parser<ImmediateValueNode> =
        TypeNode.parser <~~ spaces ~~ ImmediateNode.parser.! ^^ ImmediateValueNode.init
     .. "an immediate value"
}

extension VariableNode : Parsible {
    public static let parser: Parser<VariableNode> =
        Lexer.character("@") ~~> identifier.! ^^^ VariableNode.global
      | Lexer.character("%") ~~> identifier.! ^^^ VariableNode.temporary
      | ImmediateNode.parser.! ^^^ VariableNode.immediate
     .. "a variable"
}

extension OperandNode : Parsible {
    public static let parser: Parser<OperandNode> =
        TypeNode.parser ^^ curry(OperandNode.init)
     ** (spaces ~~> ShapeNode.parser).?
     ** (spaces.! ~~> VariableNode.parser.!)
     .. "an operand"
}

import enum DLVM.ElementwiseFunction
import enum DLVM.LogicOperator
import enum DLVM.ComparisonPredicate
import enum DLVM.AggregationFunction
import enum DLVM.ArithmeticOperator
import enum DLVM.ReductionFunction
import enum DLVM.BinaryIntegrationFunction
import protocol DLVM.LexicallyConvertible

extension Parsible where Self : LexicallyConvertible {
    public static var parser: Parser<Self> {
        return identifier.map(Self.lexicon)
    }
}

extension ElementwiseFunction : Parsible {}
extension BinaryIntegrationFunction: Parsible {}
extension ArithmeticOperator : Parsible {}
extension ComparisonPredicate : Parsible {}
extension LogicOperator: Parsible {}

extension ReductionFunction : Parsible {
    public static let parser: Parser<ReductionFunction> =
        LogicOperator.parser       ^^ ReductionFunction.logical
      | ArithmeticOperator.parser  ^^ ReductionFunction.arithmetic
     .. "reduction function"
}

extension AggregationFunction : Parsible {
    public static let parser: Parser<AggregationFunction> =
        identifier.map(AggregationFunction.lexicon)
      | Lexer.token("scan") ~~> spaces ~~> ReductionFunction.parser.! ^^ AggregationFunction.scan
}

extension LoopConditionNode : Parsible {

    private static let forTimesParser: Parser<LoopConditionNode> =
        OperandNode.parser.nonbacktracking().amid(spaces)
                          .between(Lexer.token("for"), Lexer.token("times").!)
    ^^^ LoopConditionNode.times

    private static let untilEqualParser: Parser<LoopConditionNode> =
        "until" ~~> OperandNode.parser.nonbacktracking().amid(spaces)
     ^^ curry(LoopConditionNode.untilEqual)
     ** (Lexer.token("equals") ~~> spaces ~~> OperandNode.parser.!)

    public static let parser: Parser<LoopConditionNode> = forTimesParser
                                                 | untilEqualParser
                                                .. "a loop condition"
}

extension InstructionNode : Parsible {

    private static let unaryParser: Parser<InstructionNode> =
      ( ElementwiseFunction.parser <~~ spaces ^^ curry(InstructionNode.elementwise)
      | AggregationFunction.parser <~~ spaces ^^ curry(InstructionNode.aggregate)
      | "reduce" ~~> spaces ~~> ReductionFunction.parser.! <~~ spaces
                                              ^^ curry(InstructionNode.reduce)
      | "load" ~~> spaces                     ^^= curry(InstructionNode.load)
      ) ** OperandNode.parser.!

    private static let binaryParser: Parser<InstructionNode> =
      ( BinaryIntegrationFunction.parser <~~ spaces ^^ curry(InstructionNode.binaryReduction)
      | ArithmeticOperator.parser <~~ spaces        ^^ curry(InstructionNode.arithmetic)
      | LogicOperator.parser <~~ spaces             ^^ curry(InstructionNode.logic)
      | ComparisonPredicate.parser <~~ spaces       ^^ curry(InstructionNode.comparison)
      | "mmul" ~~> spaces                           ^^= curry(InstructionNode.matrixMultiply)
      | "tmul" ~~> spaces                           ^^= curry(InstructionNode.tensorMultiply)
      ) ** OperandNode.parser.! <~~ comma.! ** OperandNode.parser.!

    private static let concatParser: Parser<InstructionNode> =
        "concat" ~~> spaces ^^= curry(InstructionNode.concatenate)
     ** OperandNode.parser.nonbacktracking().many(separatedBy: comma)
     ** (Lexer.token("along").amid(spaces) ~~> number).?

    private static let shapeCastParser: Parser<InstructionNode> =
        "shapecast" ~~> spaces ^^= curry(InstructionNode.shapeCast)
     ** OperandNode.parser.! <~~ Lexer.token("to").amid(spaces)
     ** ShapeNode.parser.!

    private static let typeCastParser: Parser<InstructionNode> =
        "typecast" ~~> spaces ^^= curry(InstructionNode.typeCast)
     ** OperandNode.parser.! <~~ Lexer.token("to").amid(spaces)
     ** TypeNode.parser.!

    private static let storeParser: Parser<InstructionNode> =
        "store" ~~> spaces ^^= curry(InstructionNode.store)
     ** OperandNode.parser.! <~~ Lexer.token("to").amid(spaces)
     ** OperandNode.parser.!

    private static let loopParser: Parser<InstructionNode> =
        "loop" ~~> spaces ^^= curry(InstructionNode.loop)
     ** BasicBlockNode.parser.! <~~ spaces.!
     ** LoopConditionNode.parser.!

    public static let parser: Parser<InstructionNode> = unaryParser
                                               | binaryParser
                                               | concatParser
                                               | shapeCastParser
                                               | typeCastParser
                                               | storeParser
                                               | loopParser
                                              .. "an instruction"
}

extension InstructionDeclarationNode : Parsible {
    public static let parser: Parser<InstructionDeclarationNode> =
        spaces.? ~~> (Lexer.character("%") ~~> identifier <~~ Lexer.character("=").amid(spaces.?)).?
     ^^ curry(InstructionDeclarationNode.init)
     ** InstructionNode.parser
}

extension BasicBlockNode : Parsible {
    private static let nonExtensionParser: Parser<BasicBlockNode> =
        Lexer.character("!") ~~> identifier.! <~~ spaces.? ^^ curry(BasicBlockNode.init)
    <~~ Lexer.character("{").amid(spaces.?).! <~~ linebreaks.!
     ** .return(nil)
     ** InstructionDeclarationNode.parser.manyOrNone(separatedBy: linebreaks)
    <~~ linebreaks <~~ Lexer.character("}").!

    private static let extensionParser: Parser<BasicBlockNode> =
        "extension" ~~> spaces ~~>
        Lexer.character("!") ~~> identifier.! <~~ spaces ^^ curry(BasicBlockNode.init)
     ** (Lexer.token("for").! ~~> spaces.! ~~> identifier.!) <~~ spaces
    <~~ Lexer.character("{").amid(spaces.?).! <~~ linebreaks.!
     ** InstructionDeclarationNode.parser.manyOrNone(separatedBy: linebreaks)
    <~~ linebreaks <~~ Lexer.character("}").!

    public static let parser: Parser<BasicBlockNode> = nonExtensionParser
                                              | extensionParser
                                             .. "a basic block"
}

extension DeclarationNode.Role : Parsible {
    public static let parser: Parser<DeclarationNode.Role> =
        Lexer.token("input")     ^^= .input
      | Lexer.token("parameter") ^^= .parameter
      | Lexer.token("output")    ^^= .output
     .. "a global variable role (input, parameter, output)"
}

extension InitializerNode : Parsible {
    public static let parser: Parser<InitializerNode> =
        ImmediateValueNode.parser ^^^ InitializerNode.immediate
      | Lexer.token("repeating") ~~> spaces ~~> ImmediateValueNode.parser.! ^^^ InitializerNode.repeating
      | Lexer.token("random") ~~> Lexer.token("from").amid(spaces) ~~>
        ImmediateValueNode.parser.! ^^ curry(InitializerNode.random)
        ** (Lexer.token("to").amid(spaces) ~~> ImmediateValueNode.parser.!)
     .. "an initializer"
}

extension DeclarationNode : Parsible {
    public static let parser: Parser<DeclarationNode> =
        Lexer.token("declare") ~~> Role.parser.amid(spaces.!) ^^ curry(DeclarationNode.init)
     ** TypeNode.parser.! <~~ spaces
     ** (ShapeNode.parser <~~ spaces).?
     ** (Lexer.character("@") ~~> identifier).! .. "an identifier"
     ** (Lexer.character("=").amid(spaces.?) ~~> InitializerNode.parser.!).?
     .. "a declaration"
}

extension ModuleNode : Parsible {
    public static let parser: Parser<ModuleNode> =
        linebreaks.? ~~> Lexer.token("module") ~~> spaces ~~> identifier.! ^^ curry(ModuleNode.init)
     ** DeclarationNode.parser.manyOrNone(separatedBy: linebreaks).amid(linebreaks.?)
     ** BasicBlockNode.parser.manyOrNone(separatedBy: linebreaks).ended(by: linebreaks.?)
}
