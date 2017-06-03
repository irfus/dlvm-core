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

import CoreTensor
import DLVM

private struct SymbolTable {
    var locals: [String : Value] = [:]
    var globals: [String : Value] = [:]
    var nominalTypes: [String : Type] = [:]
    var basicBlocks: [String : BasicBlock] = [:]
}

public class Parser {
    public let tokens: [Token]
    fileprivate lazy var restTokens: ArraySlice<Token> = ArraySlice(self.tokens)
    fileprivate var symbolTable = SymbolTable()

    public init(tokens: [Token]) {
        self.tokens = tokens
    }

    public init(text: String) throws {
        let lexer = Lexer(text: text)
        tokens = try lexer.performLexing()
    }
}

private extension Parser {

    var currentToken: Token? {
        guard let first = restTokens.first else { return nil }
        return first
    }

    var nextToken: Token? {
        return restTokens.dropFirst().first
    }

    var currentLocation: SourceLocation? {
        return currentToken?.range.lowerBound
    }

    var isEOF: Bool {
        return restTokens.isEmpty
    }

    @discardableResult
    func consumeToken() -> Token {
        return restTokens.removeFirst()
    }

    func consume(if predicate: (TokenKind) throws -> Bool) rethrows {
        guard let token = currentToken else { return }
        if try predicate(token.kind) {
            consumeToken()
        }
    }

    func consume(while predicate: (TokenKind) throws -> Bool) rethrows {
        while let first = currentToken, try predicate(first.kind) {
            consumeToken()
        }
    }

    @discardableResult
    func consumeIfAny(_ tokenKind: TokenKind) -> Token? {
        if let tok = currentToken, tok.kind == tokenKind {
            return restTokens.removeFirst()
        }
        return nil
    }

    @discardableResult
    func consume(_ tokenKind: TokenKind) throws -> Token {
        guard let first = restTokens.first else {
            throw ParseError.unexpectedEndOfInput(expected: String(describing: tokenKind))
        }
        guard first.kind == tokenKind else {
            throw ParseError.unexpectedToken(expected: String(describing: tokenKind), first)
        }
        return restTokens.removeFirst()
    }

    @discardableResult
    func consumepOrDiagnose(_ expected: String) throws -> Token {
        guard currentToken != nil else {
            throw ParseError.unexpectedEndOfInput(expected: expected)
        }
        return consumeToken()
    }

    @discardableResult
    func peekOrDiagnose(_ expected: String) throws -> Token {
        guard let tok = currentToken else {
            throw ParseError.unexpectedEndOfInput(expected: expected)
        }
        return tok
    }

    func parseInteger() throws -> (Int, SourceRange) {
        let name: String = "an integer"
        let tok = try consumepOrDiagnose(name)
        switch tok.kind {
        case let .integer(i): return (i, tok.range)
        default: throw ParseError.unexpectedToken(expected: name, tok)
        }
    }

    func parseDataType() throws -> (DataType, SourceRange) {
        let name: String = "a data type"
        let tok = try consumepOrDiagnose(name)
        switch tok.kind {
        case let .dataType(dt): return (dt, tok.range)
        default: throw ParseError.unexpectedToken(expected: name, tok)
        }
    }

    func parseIdentifier(ofKind kind: IdentifierKind, isDefinition: Bool = false) throws -> String {
        let tok = try consumepOrDiagnose("an identifier")
        let name: String
        switch tok.kind {
        case .identifier(kind, let id): name = id
        default: throw ParseError.unexpectedIdentifierKind(kind, tok)
        }
        /// If we are parsing a name definition, check for its uniqueness
        let contains: (String) -> Bool
        switch kind {
        case .basicBlock: contains = symbolTable.basicBlocks.keys.contains
        case .global: contains = symbolTable.globals.keys.contains
        case .temporary: contains = symbolTable.locals.keys.contains
        case .type: contains = symbolTable.nominalTypes.keys.contains
        default: return name
        }
        guard isDefinition ? !contains(name) : contains(name) else {
            throw ParseError.redefinedIdentifier(tok)
        }
        return name
    }

    @discardableResult
    func withBacktracking<T>(execute: () throws -> T) rethrows -> T {
        let originalTokens = restTokens
        do {
            return try execute()
        } catch let error {
            restTokens = originalTokens
            throw error
        }
    }

    @discardableResult
    func consumeWrappablePunctuation(_ punct: Punctuation) throws -> Token {
        consumeAnyNewLines()
        let tok = try consume(.punctuation(punct))
        consumeAnyNewLines()
        return tok
    }

    func consumeStringLiteral() throws -> String {
        let tok = try consumepOrDiagnose("a string literal")
        guard case let .stringLiteral(str) = tok.kind else {
            throw ParseError.unexpectedToken(expected: "a string literal", tok)
        }
        return str
    }

    func consumeAttribute() throws -> Function.Attribute {
        let tok = try consumepOrDiagnose("an attribute")
        guard case let .attribute(attr) = tok.kind else {
            throw ParseError.unexpectedToken(expected: "an attribute", tok)
        }
        return attr
    }

    func consumeAnyNewLines() {
        consume(while: {$0 == .newLine})
    }

    func consumeOneOrMore(_ kind: TokenKind) throws {
        try consume(kind)
        consume(while: { $0 == kind })
    }

    func parseMany<T>(_ parser: () throws -> T,
                      separatedBy parseSeparator: (() throws -> ())? = nil) -> [T] {
        var uses: [T] = []
        guard let first = try? parser() else { return uses }
        uses.append(first)
        while let use: T = try? withBacktracking(execute: {
            try parseSeparator?()
            return try parser()
        }) { uses.append(use) }
        return uses
    }
    
}

extension Parser {
    /// Parse one of many Uses separated by ','
    func parseUseList() -> [Use] {
        return parseMany({ try parseUse().0 },
                         separatedBy: { try self.consumeWrappablePunctuation(.comma) })
    }

    /// Parse a literal
    func parseLiteral() throws -> (Literal, SourceRange) {
        let tok = try consumepOrDiagnose("a literal")
        switch tok.kind {
        /// Float
        case let .float(f):
            return (.scalar(.float(f)), tok.range)
        /// Integer
        case let .integer(i):
            return (.scalar(.int(i)), tok.range)
        /// Boolean `true`
        case .keyword(.true):
            return (.scalar(.bool(true)), tok.range)
        /// Boolean `false`
        case .keyword(.false):
            return (.scalar(.bool(false)), tok.range)
        /// `null`
        case .keyword(.null):
            return (.null, tok.range)
        /// `undefined`
        case .keyword(.undefined):
            return (.undefined, tok.range)
        /// `zero`
        case .keyword(.zero):
            return (.zero, tok.range)
        /// Array
        case .punctuation(.leftSquareBracket):
            let elements = parseUseList()
            try consumeWrappablePunctuation(.rightSquareBracket)
            return (.array(elements), tok.range)
        /// Tuple
        case .punctuation(.leftParenthesis):
            let elements = parseUseList()
            try consumeWrappablePunctuation(.rightParenthesis)
            return (.tuple(elements), tok.range)
        /// Tensor
        case .punctuation(.leftAngleBracket):
            let elements = parseUseList()
            try consumeWrappablePunctuation(.rightAngleBracket)
            return (.tensor(elements), tok.range)
        /// Struct
        case .punctuation(.leftCurlyBracket):
            var fields: [(String, Use)] = []
            while let field: (String, Use) = try? withBacktracking(execute: {
                let key = try parseIdentifier(ofKind: .key)
                try consumeWrappablePunctuation(.colon)
                consumeAnyNewLines()
                let (val, _) = try parseUse()
                return (key, val)
            }) { fields.append(field) }
            let rightBkt = try consumeWrappablePunctuation(.rightCurlyBracket)
            return (.struct(fields), tok.startLocation..<rightBkt.endLocation)
        default:
            throw ParseError.unexpectedToken(expected: "a literal", tok)
        }
    }

    /// Parse a shape
    func parseNonScalarShape() throws -> (TensorShape, SourceRange) {
        let (first, firstRange) = try parseInteger()
        var dims = [first]
        var lastLoc = firstRange.upperBound
        while let dim: Int = try? withBacktracking(execute: {
            try consumeWrappablePunctuation(.times)
            let (num, range) = try parseInteger()
            lastLoc = range.upperBound
            return num
        }) { dims.append(dim) }
        return (TensorShape(dims), firstRange.lowerBound..<lastLoc)
    }

    func parseType() throws -> (Type, SourceRange) {
        let tok = try consumepOrDiagnose("a type")
        switch tok.kind {
        /// Void
        case .keyword(.void):
            return (.void, tok.range)
        /// Scalar
        case .dataType(let dt):
            return (.tensor([], dt), tok.range)
        /// Array
        case .punctuation(.leftSquareBracket):
            let (count, _) = try parseInteger()
            try consumeWrappablePunctuation(.times)
            let (elementType, _) = try parseType()
            let rightBkt = try consumeWrappablePunctuation(.rightSquareBracket)
            return (.array(count, elementType), tok.startLocation..<rightBkt.endLocation)
        /// Tensor
        case .punctuation(.leftAngleBracket):
            let (shape, _) = try parseNonScalarShape()
            try consumeWrappablePunctuation(.times)
            let (dt, _) = try parseDataType()
            let rightBkt = try consumeWrappablePunctuation(.rightAngleBracket)
            return (.tensor(shape, dt), tok.startLocation..<rightBkt.endLocation)
        /// Tuple
        case .punctuation(.leftParenthesis):
            let elementTypes = parseMany({
                try parseType().0
            }, separatedBy: {
                try self.consumeWrappablePunctuation(.comma)
            })
            let rightBkt = try consumeWrappablePunctuation(.rightParenthesis)
            return (.tuple(elementTypes), tok.startLocation..<rightBkt.endLocation)
        /// Nominal type
        case .identifier(.type, let typeName):
            guard let type = symbolTable.nominalTypes[typeName] else {
                throw ParseError.undefinedNominalType(tok)
            }
            return (type, tok.range)
        default:
            throw ParseError.unexpectedToken(expected: "a type", tok)
        }
    }

    func parseTypeSignature() throws -> (Type, SourceRange) {
        try consumeWrappablePunctuation(.colon)
        return try parseType()
    }

    func parseUse(in basicBlock: BasicBlock? = nil) throws -> (Use, SourceRange) {
        let tok = try peekOrDiagnose("a use of value")
        let use: Use
        let range: SourceRange
        switch tok.kind {
        /// Identifier
        case let .identifier(kind, id):
            consumeToken()
            /// Either a global or a local
            let maybeVal: Value?
            switch kind {
            case .global: maybeVal = symbolTable.globals[id]
            case .temporary: maybeVal = symbolTable.locals[id]
            default: throw ParseError.unexpectedIdentifierKind(kind, tok)
            }
            guard let val = maybeVal else {
                throw ParseError.undefinedIdentifier(tok)
            }
            use = val.makeUse()
            let (type, typeSigRange) = try parseTypeSignature()
            range = tok.startLocation..<typeSigRange.upperBound
            guard type == use.type else {
                throw ParseError.typeMismatch(expected: type, range)
            }
        /// Anonymous local identifier in a basic block
        case let .anonymousIdentifier(bbIndex, instIndex):
            guard let bb = basicBlock else {
                throw ParseError.anonymousIdentifierNotInLocal(tok)
            }
            consumeToken()
            let function = bb.parent
            /// Criteria for identifier index:
            /// - BB referred to must precede the current BB
            /// - Instruction referred to must precede the current instruction
            guard function.indices.contains(bbIndex), bbIndex <= bb.indexInParent else {
                throw ParseError.invalidAnonymousIdentifierIndex(tok)
            }
            let refBB = function[bbIndex]
            guard refBB.indices.contains(instIndex) else {
                throw ParseError.invalidAnonymousIdentifierIndex(tok)
            }
            use = %refBB[instIndex]
            let (type, typeSigRange) = try parseTypeSignature()
            range = tok.startLocation..<typeSigRange.upperBound
            guard type == use.type else {
                throw ParseError.typeMismatch(expected: type, range)
            }
        /// Literal
        case .float(_), .integer(_), .keyword(.true), .keyword(.false),
             .punctuation(.leftAngleBracket),
             .punctuation(.leftCurlyBracket),
             .punctuation(.leftSquareBracket),
             .punctuation(.leftParenthesis):
            let (lit, _) = try parseLiteral()
            let (type, typeSigRange) = try parseTypeSignature()
            range = tok.startLocation..<typeSigRange.upperBound
            use = .literal(type, lit)
        default:
            throw ParseError.unexpectedToken(expected: "a use of value", tok)
        }
        return (use, range)
    }

    func parseInstructionKind() throws -> InstructionKind {
        let tok = try consumepOrDiagnose("an opcode")
        guard case let .opcode(opcode) = tok.kind else {
            throw ParseError.unexpectedToken(expected: "an opcode", tok)
        }
        /// - todo: parse instruction kind
        switch opcode {
        /// 'branch' <bb> '(' (<val> (',' <val>)*)? ')'
        case .branch: break
        /// 'conditional' <bb> 'then' <bb> '(' (<val> (',' <val>)*)? ')'
        ///                    'else' <bb> '(' (<val> (',' <val>)*)? ')'
        case .conditional: break
        /// 'return' <val>?
        case .return: break
        /// 'dataTypeCast' <val> 'to' <data_type>
        case .dataTypeCast: break
        /// 'scan' <val> 'by' <func|assoc_op> 'along' <num> (',' <num>)*
        case .scan: break
        /// 'reduce' <val> 'by' <func|assoc_op> 'along' <num> (',' <num>)*
        case .reduce: break
        /// 'matrixMultiply' <val> ',' <val>
        case .matrixMultiply: break
        /// 'concatenate' <val> (',' <val>)* along <num>
        case .concatenate: break
        /// 'transpose' <val>
        case .transpose: break
        /// 'shapeCast' <val> 'to' <shape>
        case .shapeCast: break
        /// 'bitCast' <val> 'to' <type>
        case .bitCast: break
        /// 'extract' <num|key|val> (',' <num|key|val>)* 'from' <val>
        case .extract: break
        /// 'insert' <val> 'to' <val> 'at' <num|key|val> (',' <num|key|val>)*
        case .insert: break
        /// 'apply' <val> '(' <val>+ ')' 
        case .apply: break
        /// 'allocateStack' <type> 'count' <num>
        case .allocateStack: break
        /// 'allocateHeap' <type> 'count' <val>
        case .allocateHeap: break
        /// 'allocateBox' <type>
        case .allocateBox: break
        /// 'projectBox' <val>
        case .projectBox: break
        /// 'retain' <val>
        case .retain: break
        /// 'release' <val>
        case .release: break
        /// 'deallocate' <val>
        case .deallocate: break
        /// 'load' <val>
        case .load: break
        /// 'store' <val> 'to' <val>
        case .store: break
        /// 'elementPointer' <val> 'at' <num|key|val> (<num|key|val> ',')*
        case .elementPointer: break
        /// 'copy' 'from' <val> 'to' <val> 'count' <val>
        case .copy: break
        case .trap: break
        /// <binary_op> <val>, <val> (broadcast [left|right] <num> (',' <num>)*)?
        case .binaryOp(_): break
        /// <unary_op> <val>
        case .unaryOp(_): break
        }
        fatalError("Unimplemented")
    }

    func parseInstruction(in basicBlock: BasicBlock) throws -> Instruction {
        /// Instruction must start with an indentation
        try consume(.indent)
        let tok = try peekOrDiagnose("an instruction or a local identifier")
        let inst: Instruction
        switch tok.kind {
        case let .identifier(.temporary, name):
            consumeToken()
            try consumeWrappablePunctuation(.equal)
            let kind = try parseInstructionKind()
            guard !symbolTable.locals.keys.contains(name) else {
                throw ParseError.redefinedIdentifier(tok)
            }
            inst = Instruction(name: name, kind: kind, parent: basicBlock)
            symbolTable.locals[name] = inst
        case let .anonymousIdentifier(bbIndex, instIndex):
            /// Check BB index and instruction index
            /// - BB index must equal the current BB index
            /// - Instruction index must equal the next instruction index, 
            ///   i.e. the current instruction count
            guard bbIndex == basicBlock.indexInParent, instIndex == basicBlock.endIndex else {
                throw ParseError.invalidAnonymousIdentifierIndex(tok)
            }
            consumeToken()
            try consumeWrappablePunctuation(.equal)
            let kind = try parseInstructionKind()
            inst = Instruction(kind: kind, parent: basicBlock)
        default:
            let kind = try parseInstructionKind()
            inst = Instruction(kind: kind, parent: basicBlock)
        }
        try consumeOneOrMore(.newLine)
        basicBlock.append(inst)
        return inst
    }

    func parseBasicBlock(in function: Function) throws -> BasicBlock {
        /// Parse basic block header
        let name = try parseIdentifier(ofKind: .basicBlock, isDefinition: true)
        try consumeWrappablePunctuation(.leftParenthesis)
        let args: [(String, Type)] = parseMany({
            let name = try parseIdentifier(ofKind: .temporary, isDefinition: true)
            let (type, _) = try parseTypeSignature()
            return (name, type)
        }, separatedBy: {
            try self.consumeWrappablePunctuation(.comma)
        })
        try consumeWrappablePunctuation(.rightParenthesis)
        try consume(.punctuation(.colon))
        try consumeOneOrMore(.newLine)
        let bb = BasicBlock(name: name, arguments: args, parent: function)
        /// Insert args into symbol table
        for arg in bb.arguments {
            symbolTable.locals[arg.name] = arg
        }
        /// Parse instructions
        _ = parseMany({
            try parseInstruction(in: bb)
        }, separatedBy: {
            try self.consumeOneOrMore(.newLine)
        })
        /// Insert BB into symbol table
        symbolTable.basicBlocks[name] = bb
        return bb
    }

    func parseIndexList() throws -> [Int] {
        return parseMany({
            try parseInteger().0
        }, separatedBy: {
            try self.consumeWrappablePunctuation(.comma)
        })
    }

    func parseFunction(in module: Module) throws -> Function {
        try consume(.keyword(.func))
        let name = try parseIdentifier(ofKind: .global, isDefinition: true)
        let (type, typeSigRange) = try parseTypeSignature()
        guard case let .function(args, ret) = type.canonical else {
            throw ParseError.notFunctionType(typeSigRange)
        }
        let attributes = parseMany({ try consumeAttribute() })
        /// - TODO: parse declaration kind
        let function = Function(name: name, argumentTypes: args, returnType: ret,
                                attributes: Set(attributes), parent: module)
        /// Insert function to symbol table
        symbolTable.globals[name] = function
        try consumeWrappablePunctuation(.leftCurlyBracket)
        _ = parseMany({
            try parseBasicBlock(in: function)
        }, separatedBy: {
            try self.consumeOneOrMore(.newLine)
        })
        try consume(.punctuation(.rightCurlyBracket))
        return function
    }
}

public extension Parser {
    /// Parser entry
    func parseModule() throws -> Module {
        consumeAnyNewLines()
        try consume(.keyword(.module))
        let name = try consumeStringLiteral()
        let module = Module(name: name)
        
        /// ... end of input
        return module
    }
}
