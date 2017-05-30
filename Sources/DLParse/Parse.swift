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

public class Parser {
    public fileprivate(set) var tokens: ArraySlice<Token>

    public init(tokens: [Token]) {
        self.tokens = ArraySlice(tokens)
    }

    public init(text: String) throws {
        let lexer = Lexer(text: text)
        tokens = try ArraySlice(lexer.performLexing())
    }
}

private extension Parser {

    var currentToken: Token? {
        guard let first = tokens.first else { return nil }
        return first
    }

    var nextToken: Token? {
        return tokens.dropFirst().first
    }

    var currentLocation: SourceLocation? {
        return currentToken?.range.lowerBound
    }

    var isEOF: Bool {
        return tokens.isEmpty
    }
    
    func consume(if predicate: (TokenKind) throws -> Bool) rethrows {
        guard let token = tokens.first else { return }
        if try predicate(token.kind) {
            tokens.removeFirst()
        }
    }

    func consume(while predicate: (TokenKind) throws -> Bool) rethrows {
        while let first = tokens.first, try predicate(first.kind) {
            tokens.removeFirst()
        }
    }

    @discardableResult
    func consumeIfAny(_ tokenKind: TokenKind) -> Token? {
        if let tok = tokens.first, tok.kind == tokenKind {
            return tokens.removeFirst()
        }
        return nil
    }

    @discardableResult
    func consume(_ tokenKind: TokenKind) throws -> Token {
        guard let first = tokens.first else {
            throw ParseError.unexpectedEndOfInput(expected: String(describing: tokenKind))
        }
        guard first.kind == tokenKind else {
            throw ParseError.unexpectedToken(expected: String(describing: tokenKind), first)
        }
        return tokens.removeFirst()
    }

    @discardableResult
    func consumeToken() -> Token {
        return tokens.removeFirst()
    }

    @discardableResult
    func consumeOrDiagnose(_ expected: String) throws -> Token {
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

    func parseIdentifier(ofKind kind: IdentifierKind) throws -> String {
        let tok = try consumeOrDiagnose("an identifier")
        switch tok.kind {
        case .identifier(kind, let id): return id
        default: throw ParseError.unexpectedIdentifierKind(kind, tok)
        }
    }

    @discardableResult
    func withBacktracking<T>(execute: () throws -> T) rethrows -> T {
        let originalTokens = tokens
        do {
            return try execute()
        } catch let error {
            tokens = originalTokens
            throw error
        }
    }

    func consumeAnyNewLines() {
        consume(while: {$0 == .newLine})
    }

    func parseMany<T>(_ parser: () throws -> T,
                      separatedBy: () throws -> ()) -> [T] {
        var uses: [T] = []
        guard let first = try? parser() else { return uses }
        uses.append(first)
        while let use: T = try? withBacktracking(execute: {
            try consume(.punctuation(.comma))
            return try parser()
        }) { uses.append(use) }
        return uses
    }
    
}

extension Parser {
    /// Parse one of many Uses separated by ','
    func parseUseList() -> [UseNode] {
        return parseMany({ try parseUse() },
                         separatedBy: { try consume(.punctuation(.comma)) })
    }

    /// Parse a literal
    func parseLiteral() throws -> LiteralNode {
        let tok = try consumeOrDiagnose("a literal")
        switch tok.kind {
        /// Float
        case let .float(f):
            return .scalar(.float(f))
        /// Integer
        case let .integer(i):
            return .scalar(.int(i))
        /// Boolean `true`
        case .keyword(.true):
            return .scalar(.bool(true))
        /// Boolean `false`
        case .keyword(.false):
            return .scalar(.bool(false))
        /// `null`
        case .keyword(.null):
            return .null
        /// `undefined`
        case .keyword(.undefined):
            return .undefined
        /// `zero`
        case .keyword(.zero):
            return .zero
        /// Array
        case .punctuation(.leftSquareBracket):
            let elements = parseUseList()
            try consume(.punctuation(.rightSquareBracket))
            return .array(elements)
        /// Tuple
        case .punctuation(.leftParenthesis):
            let elements = parseUseList()
            try consume(.punctuation(.rightParenthesis))
            return .tuple(elements)
        /// Tensor
        case .punctuation(.leftAngleBracket):
            let elements = parseUseList()
            try consume(.punctuation(.rightAngleBracket))
            return .tensor(elements)
        /// Struct
        case .punctuation(.leftCurlyBracket):
            var fields: [(String, UseNode)] = []
            while let field: (String, UseNode) = try? withBacktracking(execute: {
                let key = try parseIdentifier(ofKind: .key)
                try consume(.punctuation(.colon))
                consumeAnyNewLines()
                let val = try parseUse()
                return (key, val)
            }) { fields.append(field) }
            try consume(.punctuation(.rightCurlyBracket))
            return .struct(fields)
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
            try consume(.punctuation(.times))
            let (num, range) = try parseInteger()
            lastLoc = range.upperBound
            return num
        }) { dims.append(dim) }
        return (TensorShape(dims), firstRange.lowerBound..<lastLoc)
    }

    func parseType() throws -> TypeNode {
        let tok = try consumeOrDiagnose("a type")
        switch tok.kind {
        /// Void
        case .keyword(.void):
            return .void(tok.range)
        /// Scalar
        case .dataType(let dt):
            return .tensor([], dt, tok.range)
        /// Array
        case .punctuation(.leftSquareBracket):
            let (count, _) = try parseInteger()
            try consume(.punctuation(.times))
            let elementType = try parseType()
            let rightBkt = try consume(.punctuation(.rightSquareBracket))
            return .array(count, elementType, tok.startLocation..<rightBkt.endLocation)
        /// Tensor
        case .punctuation(.leftAngleBracket):
            let (shape, _) = try parseNonScalarShape()
            try consume(.punctuation(.times))
            let (dt, _) = try parseDataType()
            let rightBkt = try consume(.punctuation(.rightAngleBracket))
            return .tensor(shape, dt, tok.startLocation..<rightBkt.endLocation)
        /// Tuple
        case .punctuation(.leftParenthesis):
            let elementTypes = parseMany({
                try parseType()
            }, separatedBy: {
                try consume(.punctuation(.comma))
            })
            let rightBkt = try consume(.punctuation(.rightParenthesis))
            return .tuple(elementTypes, tok.startLocation..<rightBkt.endLocation)
        /// Nominal type
        case .identifier(.type, let typeName):
            return .nominal(typeName, tok.range)
        default:
            throw ParseError.unexpectedToken(expected: "a type", tok)
        }
    }

    func parseUse() throws -> UseNode {
        let tok = try peekOrDiagnose("a use of value")
        let useKind: UseNode.Kind
        switch tok.kind {
        /// Identifier
        case let .identifier(kind, id):
            consumeToken()
            /// Either a global or a local
            switch kind {
            case .global: useKind = .global(id)
            case .temporary: useKind = .temporary(id)
            default: throw ParseError.unexpectedIdentifierKind(kind, tok)
            }
        /// Unambiguous literals
        case .float(_), .integer(_), .keyword(.true), .keyword(.false),
             .punctuation(.leftAngleBracket),
             .punctuation(.leftCurlyBracket),
             .punctuation(.leftSquareBracket):
            useKind = try .literal(parseLiteral())
        /// Tuple literal OR constant expression
        case .punctuation(.leftParenthesis):
            if let next = nextToken, next.kind.isOpcode {
                /// Constant expression
                consumeToken()
                useKind = try .constant(parseInstructionKind())
                try consume(.punctuation(.rightParenthesis))
            } else {
                useKind = try .literal(parseLiteral())
            }
        default:
            throw ParseError.unexpectedToken(expected: "a use of value", tok)
        }
        try consume(.punctuation(.colon))
        consumeAnyNewLines()
        let type = try parseType()
        return UseNode(type: type, kind: useKind,
                       range: tok.startLocation..<type.range.upperBound)
    }

    func parseInstructionKind() throws -> InstructionKind {
        let tok = try consumeOrDiagnose("an opcode")
        guard case let .opcode(opcode) = tok.kind else {
            throw ParseError.unexpectedToken(expected: "an opcode", tok)
        }
        /// - todo: parse instruction kind
        switch opcode {
        case .branch: break
        case .conditional: break
        case .return: break
        case .dataTypeCast: break
        case .scan: break
        case .reduce: break
        case .matrixMultiply: break
        case .concatenate: break
        case .transpose: break
        case .shapeCast: break
        case .bitCast: break
        case .extract: break
        case .insert: break
        case .apply: break
        case .gradient: break
        case .allocateStack: break
        case .allocateHeap: break
        case .allocateBox: break
        case .projectBox: break
        case .retain: break
        case .release: break
        case .deallocate: break
        case .load: break
        case .store: break
        case .elementPointer: break
        case .copy: break
        case .trap: break
        case .binaryOp(_): break
        case .unaryOp(_): break
        }
        fatalError("Unimplemented")
    }
}
