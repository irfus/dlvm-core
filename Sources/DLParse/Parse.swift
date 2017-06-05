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
    func consumeOrDiagnose(_ expected: String) throws -> Token {
        guard currentToken != nil else {
            throw ParseError.unexpectedEndOfInput(expected: expected)
        }
        return consumeToken()
    }

    @discardableResult
    func withPeekedToken<T>(_ expected: String, _ execute: (Token) throws -> T?) throws -> T {
        let tok = try peekOrDiagnose(expected)
        guard let result = try execute(tok) else {
            throw ParseError.unexpectedToken(expected: expected, tok)
        }
        return result
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
        let tok = try consumeOrDiagnose(name)
        switch tok.kind {
        case let .integer(i): return (i, tok.range)
        default: throw ParseError.unexpectedToken(expected: name, tok)
        }
    }

    func parseDataType() throws -> (DataType, SourceRange) {
        let name: String = "a data type"
        let tok = try consumeOrDiagnose(name)
        switch tok.kind {
        case let .dataType(dt): return (dt, tok.range)
        default: throw ParseError.unexpectedToken(expected: name, tok)
        }
    }

    func parseIdentifier(ofKind kind: IdentifierKind, isDefinition: Bool = false) throws -> (String, Token) {
        let tok = try consumeOrDiagnose("an identifier")
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
        default: return (name, tok)
        }
        guard isDefinition ? !contains(name) : contains(name) else {
            throw ParseError.redefinedIdentifier(tok)
        }
        return (name, tok)
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

    func withPreservedState(execute: () throws -> ()) {
        let originalTokens = restTokens
        _ = try? execute()
        restTokens = originalTokens
    }

    @discardableResult
    func consumeWrappablePunctuation(_ punct: Punctuation) throws -> Token {
        consumeAnyNewLines()
        let tok = try consume(.punctuation(punct))
        consumeAnyNewLines()
        return tok
    }

    func consumeStringLiteral() throws -> String {
        let tok = try consumeOrDiagnose("a string literal")
        guard case let .stringLiteral(str) = tok.kind else {
            throw ParseError.unexpectedToken(expected: "a string literal", tok)
        }
        return str
    }

    func consumeAttribute() throws -> Function.Attribute {
        let tok = try consumeOrDiagnose("an attribute")
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
        let tok = try consumeOrDiagnose("a literal")
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
            let fields: [(String, Use)] = parseMany({
                let (key, _) = try parseIdentifier(ofKind: .key)
                try consumeWrappablePunctuation(.equal)
                let (val, _) = try parseUse()
                return (key, val)
            }, separatedBy: {
                try self.consumeWrappablePunctuation(.comma)
            })
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

    func parseElementKey() throws -> (ElementKey, SourceRange) {
        return try withPeekedToken("an element key", { tok in
            consumeToken()
            switch tok.kind {
            case let .integer(i):
                return (.index(i), tok.range)
            case let .identifier(.key, nameKey):
                return (.name(nameKey), tok.range)
            default:
                let (val, range) = try parseUse()
                return (.value(val), range)
            }
        })
    }

    func parseType() throws -> (Type, SourceRange) {
        let tok = try consumeOrDiagnose("a type")
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
        let opcode: InstructionKind.Opcode = try withPeekedToken("an opcode") { tok in
            guard case let .opcode(opcode) = tok.kind else {
                return nil
            }
            consumeToken()
            return opcode
        }
        /// - todo: parse instruction kind
        switch opcode {
        /// 'branch' <bb> '(' (<val> (',' <val>)*)? ')'
        case .branch:
            let (bbName, bbTok) = try parseIdentifier(ofKind: .basicBlock)
            guard let bb = symbolTable.basicBlocks[bbName] else {
                throw ParseError.undefinedIdentifier(bbTok)
            }
            try consume(.punctuation(.leftParenthesis))
            let args = parseUseList()
            try consume(.punctuation(.rightParenthesis))
            return .branch(bb, args)

        /// 'conditional' <cond> 'then' <bb> '(' (<val> (',' <val>)*)? ')'
        ///                      'else' <bb> '(' (<val> (',' <val>)*)? ')'
        case .conditional:
            let (cond, _) = try parseUse()
            /// Then
            try consume(.keyword(.then))
            let (thenBBName, thenBBTok) = try parseIdentifier(ofKind: .basicBlock)
            guard let thenBB = symbolTable.basicBlocks[thenBBName] else {
                throw ParseError.undefinedIdentifier(thenBBTok)
            }
            try consume(.punctuation(.leftParenthesis))
            let thenArgs = parseUseList()
            try consume(.punctuation(.rightParenthesis))
            /// Else
            try consume(.keyword(.else))
            let (elseBBName, elseBBTok) = try parseIdentifier(ofKind: .basicBlock)
            guard let elseBB = symbolTable.basicBlocks[elseBBName] else {
                throw ParseError.undefinedIdentifier(elseBBTok)
            }
            try consume(.punctuation(.leftParenthesis))
            let elseArgs = parseUseList()
            try consume(.punctuation(.rightParenthesis))
            return .conditional(cond, thenBB, thenArgs, elseBB, elseArgs)
            
        /// 'return' <val>?
        case .return:
            return .return(try? parseUse().0)

        /// 'dataTypeCast' <val> 'to' <data_type>
        case .dataTypeCast:
            let (srcVal, _) = try parseUse()
            try consume(.keyword(.to))
            let (dtype, _) = try parseDataType()
            return .dataTypeCast(srcVal, dtype)
            
        /// 'scan' <val> 'by' <func|assoc_op> 'along' <num> (',' <num>)*
        case .scan:
            let (val, _) = try parseUse()
            try consume(.keyword(.by))
            let combinator: ReductionCombinator = try withPeekedToken("a function or an associative operator", { tok in
                switch tok.kind {
                case .identifier(_):
                    return try .function(parseUse().0)
                case .opcode(.binaryOp(.associative(let op))):
                    return .op(op)
                default:
                    return nil
                }
            })
            try consume(.keyword(.along))
            let (firstDim, _) = try parseInteger()
            let restDims: [Int] = parseMany({
                try consumeWrappablePunctuation(.comma)
                return try parseInteger().0
            })
            return .scan(combinator, val, [firstDim] + restDims)

        /// 'reduce' <val> 'by' <func|assoc_op> 'along' <num> (',' <num>)*
        case .reduce:
            let (val, _) = try parseUse()
            try consume(.keyword(.by))
            let combinator: ReductionCombinator = try withPeekedToken("a function or an associative operator", { tok in
                switch tok.kind {
                case .identifier(_):
                    return try .function(parseUse().0)
                case .opcode(.binaryOp(.associative(let op))):
                    return .op(op)
                default:
                    return nil
                }
            })
            try consume(.keyword(.along))
            let (firstDim, _) = try parseInteger()
            let restDims: [Int] = parseMany({
                try consumeWrappablePunctuation(.comma)
                return try parseInteger().0
            })
            return .reduce(combinator, val, [firstDim] + restDims)

        /// 'matrixMultiply' <val> ',' <val>
        case .matrixMultiply:
            let (lhs, _) = try parseUse()
            try consumeWrappablePunctuation(.comma)
            let (rhs, _) = try parseUse()
            return .matrixMultiply(lhs, rhs)
            
        /// 'concatenate' <val> (',' <val>)* along <num>
        case .concatenate:
            let (firstVal, _) = try parseUse()
            let restVals: [Use] = parseMany({
                try consumeWrappablePunctuation(.comma)
                return try parseUse().0
            })
            try consume(.keyword(.along))
            let (axis, _) = try parseInteger()
            return .concatenate([firstVal] + restVals, axis: axis)

        /// 'transpose' <val>
        case .transpose:
            return try .transpose(parseUse().0)

        /// 'shapeCast' <val> 'to' <shape>
        case .shapeCast:
            let (val, _) = try parseUse()
            try consume(.keyword(.to))
            let shape: TensorShape = try withPeekedToken("dimensions separated by 'x', or 'scalar'", { tok in
                switch tok.kind {
                case .keyword(.scalar):
                    return .scalar
                case .integer(_):
                    return try parseNonScalarShape().0
                default:
                    return nil
                }
            })
            return .shapeCast(val, shape)
            
        /// 'bitCast' <val> 'to' <type>
        case .bitCast:
            let (val, _) = try parseUse()
            try consume(.keyword(.to))
            let (type, _) = try parseType()
            return .bitCast(val, type)
            
        /// 'extract' <num|key|val> (',' <num|key|val>)* 'from' <val>
        case .extract:
            let (firstKey, _) = try parseElementKey()
            let restKeys: [ElementKey] = parseMany({
                try consumeWrappablePunctuation(.comma)
                return try parseElementKey().0
            })
            try consume(.keyword(.from))
            return .extract(from: try parseUse().0, at: [firstKey] + restKeys)

        /// 'insert' <val> 'to' <val> 'at' <num|key|val> (',' <num|key|val>)*
        case .insert:
            let (srcVal, _) = try parseUse()
            try consume(.keyword(.to))
            let (destVal, _) = try parseUse()
            try consume(.keyword(.at))
            let (firstKey, _) = try parseElementKey()
            let restKeys: [ElementKey] = parseMany({
                try consumeWrappablePunctuation(.comma)
                return try parseElementKey().0
            })
            return .insert(srcVal, to: destVal, at: [firstKey] + restKeys)

        /// 'apply' <val> '(' <val>+ ')'
        case .apply:
            let (funcVal, _) = try parseUse()
            try consume(.punctuation(.leftParenthesis))
            let args = parseUseList()
            try consume(.punctuation(.rightParenthesis))
            return .apply(funcVal, args)

        /// 'allocateStack' <type> 'count' <num>
        case .allocateStack:
            let (type, _) = try parseType()
            try consume(.keyword(.count))
            let (count, _) = try parseInteger()
            return .allocateStack(type, count)
            
        /// 'allocateHeap' <type> 'count' <val>
        case .allocateHeap:
            let (type, _) = try parseType()
            try consume(.keyword(.count))
            let (count, _) = try parseUse()
            return .allocateHeap(type, count: count)
            
        /// 'allocateBox' <type>
        case .allocateBox:
            return try .allocateBox(parseType().0)

        /// 'projectBox' <val>
        case .projectBox:
            return try .projectBox(parseUse().0)

        /// 'retain' <val>
        case .retain:
            return try .retain(parseUse().0)

        /// 'release' <val>
        case .release:
            return try .release(parseUse().0)

        /// 'deallocate' <val>
        case .deallocate:
            return try .deallocate(parseUse().0)

        /// 'load' <val>
        case .load:
            return try .load(parseUse().0)

        /// 'store' <val> 'to' <val>
        case .store:
            let (src, _) = try parseUse()
            try consume(.keyword(.to))
            let (dest, _) = try parseUse()
            return .store(src, to: dest)
            
        /// 'elementPointer' <val> 'at' <num|key|val> (<num|key|val> ',')*
        case .elementPointer:
            let (base, _) = try parseUse()
            try consume(.keyword(.at))
            let (firstKey, _) = try parseElementKey()
            let restKeys: [ElementKey] = parseMany({
                try consumeWrappablePunctuation(.comma)
                return try parseElementKey().0
            })
            return .elementPointer(base, [firstKey] + restKeys)

        /// 'copy' 'from' <val> 'to' <val> 'count' <val>
        case .copy:
            try consume(.keyword(.from))
            let (src, _) = try parseUse()
            try consume(.keyword(.to))
            let (dest, _) = try parseUse()
            try consume(.keyword(.count))
            let (count, _) = try parseUse()
            return .copy(from: src, to: dest, count: count)
            
        case .trap:
            return .trap
            
        /// <binary_op> <val>, <val>
        case let .binaryOp(op):
            let (lhs, _) = try parseUse()
            try consumeWrappablePunctuation(.comma)
            let (rhs, _) = try parseUse()
            return .zipWith(op, lhs, rhs)

        /// <unary_op> <val>
        case let .unaryOp(op):
            return try .map(op, parseUse().0)
        }
    }

    func parseInstruction(in basicBlock: BasicBlock) throws -> Instruction {
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
        let (name, _) = try parseIdentifier(ofKind: .basicBlock, isDefinition: true)
        try consumeWrappablePunctuation(.leftParenthesis)
        let args: [(String, Type)] = parseMany({
            let (name, _) = try parseIdentifier(ofKind: .temporary, isDefinition: true)
            let (type, _) = try parseTypeSignature()
            return (name, type)
        }, separatedBy: {
            try self.consumeWrappablePunctuation(.comma)
        })
        try consumeWrappablePunctuation(.rightParenthesis)
        try consume(.punctuation(.colon))
        try consumeOneOrMore(.newLine)
        /// Retrieve previously added BB during scanning
        guard let bb = symbolTable.basicBlocks[name] else {
            preconditionFailure("Should've been added during the symbol scanning stage")
        }
        /// Mutate BB's args
        bb.arguments = OrderedSet(args.lazy.map(Argument.init))
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
        /// - TODO: parse declaration kind
        let startLoc = try consume(.keyword(.func)).startLocation
        let (name, _) = try parseIdentifier(ofKind: .global, isDefinition: true)
        let (type, typeSigRange) = try parseTypeSignature()
        guard case let .function(args, ret) = type.canonical else {
            throw ParseError.notFunctionType(typeSigRange)
        }
        let attributes = parseMany({ try consumeAttribute() })
        /// Retrieve previous added function during scanning
        guard let function = symbolTable.globals[name] as? Function else {
            preconditionFailure("Should've been added during the symbol scanning stage")
        }
        /// Complete function's properties
        function.argumentTypes = args
        function.returnType = ret
        function.attributes = Set(attributes)
        /// Parse definition in `{...}` when it's not a declaration
        if function.isDefinition {
            try consumeWrappablePunctuation(.leftCurlyBracket)
            _ = parseMany({
                try parseBasicBlock(in: function)
            }, separatedBy: {
                try self.consumeOneOrMore(.newLine)
            })
            try consume(.punctuation(.rightCurlyBracket))
            return function
        }
        /// Otherwise if `{` follows the declaration, emit proper diagnostics
        else if let tok = currentToken, tok.kind == .punctuation(.leftCurlyBracket) {
            throw ParseError.declarationCannotHaveBody(declaration: startLoc..<typeSigRange.upperBound,
                                                       body: tok)
        }
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
