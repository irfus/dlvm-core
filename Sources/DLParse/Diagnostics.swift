//
//  ParseError.swift
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

import DLVM

public enum LexicalError : Error {
    case unexpectedToken(SourceLocation)
    case illegalNumber(SourceRange)
    case illegalIdentifier(SourceRange)
    case invalidEscapeCharacter(UnicodeScalar, SourceLocation)
    case unclosedStringLiteral(SourceRange)
    case expectingIdentifierName(SourceLocation)
    case invalidAnonymousLocalIdentifier(SourceLocation)
    case invalidBasicBlockIndex(SourceLocation)
    case invalidInstructionIndex(SourceLocation)
    case unknownAttribute(SourceRange)
}

public enum ParseError : Error {
    case unexpectedIdentifierKind(IdentifierKind, Token)
    case unexpectedEndOfInput(expected: String)
    case unexpectedToken(expected: String, Token)
    case noDimensionsInTensorShape(Token)
    case undefinedIdentifier(Token)
    case typeMismatch(expected: Type, SourceRange)
    case undefinedNominalType(Token)
    case redefinedIdentifier(Token)
    case anonymousIdentifierNotInLocal(Token)
    case invalidAnonymousIdentifierIndex(Token)
    case notFunctionType(SourceRange)
    case invalidAttributeArguments(SourceLocation)
}

public extension LexicalError {
    var location: SourceLocation? {
        switch self {
        case .expectingIdentifierName(let loc),
             .invalidEscapeCharacter(_, let loc),
             .unexpectedToken(let loc),
             .invalidBasicBlockIndex(let loc),
             .invalidInstructionIndex(let loc),
             .invalidAnonymousLocalIdentifier(let loc):
            return loc
        case .illegalIdentifier(let range),
             .illegalNumber(let range),
             .unclosedStringLiteral(let range),
             .unknownAttribute(let range):
            return range.lowerBound
        }
    }
}

public extension ParseError {
    /// Location of the error, nil if EOF
    var location: SourceLocation? {
        switch self {
        case .unexpectedEndOfInput(_):
            return nil
        case let .unexpectedIdentifierKind(_, tok),
             let .unexpectedToken(_, tok),
             let .noDimensionsInTensorShape(tok),
             let .undefinedIdentifier(tok),
             let .undefinedNominalType(tok),
             let .redefinedIdentifier(tok),
             let .anonymousIdentifierNotInLocal(tok),
             let .invalidAnonymousIdentifierIndex(tok):
            return tok.startLocation
        case let .typeMismatch(_, range),
             let .notFunctionType(range):
            return range.lowerBound
        case let .invalidAttributeArguments(loc):
            return loc
        }
    }
}

extension ParseError : CustomStringConvertible {
    public var description : String {
        var desc = "Error at "
        if let location = location {
            desc += location.description
        } else {
            desc += "the end of file"
        }
        desc += ": "
        switch self {
        case let .unexpectedIdentifierKind(kind, tok):
            return "identifier \(tok) has unexpected kind \(kind)"
        case let .unexpectedEndOfInput(expected: expected):
            return "expected \(expected) but reached the end of input"
        case let .unexpectedToken(expected: expected, tok):
            return "expected \(expected) but found the token \(tok)"
        case let .noDimensionsInTensorShape(tok):
            return "no dimensions in tensor type at \(tok.startLocation). If you'd like it to be a scalar, use the data type (e.g. f32) directly."
        case let .undefinedIdentifier(tok):
            return "undefined identifier \(tok)"
        case let .typeMismatch(expected: ty, range):
            return "value at \(range) should have type \(ty)"
        case let .undefinedNominalType(tok):
            return "nominal type \(tok) is undefined"
        case let .redefinedIdentifier(tok):
            return "identifier \(tok) is redefined"
        case let .anonymousIdentifierNotInLocal(tok):
            return "anonymous identifier \(tok) is not in a local (basic block) context"
        case let .invalidAnonymousIdentifierIndex(tok):
            return "anonymous identifier \(tok) has invalid index"
        case let .notFunctionType(range):
            return "type signature at \(range) is not a function type"
        case .invalidAttributeArguments(_):
            return "invalid attribute arguments"
        }
    }
}

extension Token : CustomStringConvertible {
    public var description: String {
        return kind.description
    }
}

extension LexicalError : CustomStringConvertible {
    public var description: String {
        var desc = "Error at "
        if let location = location {
            desc += location.description
        } else {
            desc += "the end of file"
        }
        desc += ": "
        switch self {
        case let .illegalIdentifier(range):
            return "illegal identifier at \(range)"
        case let .illegalNumber(range):
            return "illegal number at \(range)"
        case .unexpectedToken(_):
            return "unexpected token"
        case let .invalidEscapeCharacter(char, _):
            return "invalid escape character '\(char)'"
        case let .unclosedStringLiteral(range):
            return "string literal at \(range) is not terminated"
        case .expectingIdentifierName(_):
            return "expecting identifier name"
        case .invalidAnonymousLocalIdentifier(_):
            return "invalid anonymous loacl identifier. It should look like %<bb_index>.<inst_index>, e.g. %0.1"
        case .invalidBasicBlockIndex(_):
            return "invalid index for basic block in anonymous local identifier"
        case .invalidInstructionIndex(_):
            return "invalid index for instruction in anonymous local identifier"
        case let .unknownAttribute(range):
            return "unknown attribute at \(range)"
        }
    }
}

extension Punctuation : CustomStringConvertible {
    public var description: String {
        switch self {
        case .colon: return ":"
        case .comma: return ","
        case .equal: return "="
        case .leftAngleBracket: return "<"
        case .rightAngleBracket: return ">"
        case .leftCurlyBracket: return "{"
        case .rightCurlyBracket: return "}"
        case .leftSquareBracket: return "["
        case .rightSquareBracket: return "]"
        case .leftParenthesis: return "("
        case .rightParenthesis: return ")"
        case .rightArrow: return "->"
        case .star: return "*"
        case .times: return "x"
        }
    }
}

extension InstructionKind.Opcode : CustomStringConvertible {
    public var description: String {
        switch self {
        case .branch: return "branch"
        case .conditional: return "condition"
        case .return: return "return"
        case .dataTypeCast: return "dataTypeCast"
        case .scan: return "scan"
        case .reduce: return "reduce"
        case .matrixMultiply: return "matrixMultiply"
        case .concatenate: return "concatenate"
        case .transpose: return "transpose"
        case .shapeCast: return "shapeCast"
        case .bitCast: return "bitCast"
        case .extract: return "extract"
        case .insert: return "insert"
        case .apply: return "apply"
        case .allocateStack: return "allocateStack"
        case .allocateHeap: return "allocateHeap"
        case .allocateBox: return "allocateBox"
        case .projectBox: return "projectBox"
        case .retain: return "retain"
        case .release: return "release"
        case .deallocate: return "deallocate"
        case .load: return "load"
        case .store: return "store"
        case .elementPointer: return "elementPointer"
        case .copy: return "copy"
        case .trap: return "trap"
        case let .binaryOp(op): return String(describing: op)
        case let .unaryOp(op): return String(describing: op)
        }
    }
}

extension TokenKind : CustomStringConvertible {
    public var description: String {
        switch self {
        case let .punctuation(p): return "'\(p)'"
        case let .dataType(dt): return String(describing: dt)
        case let .float(val): return val.description
        case let .integer(val): return val.description
        case let .keyword(kw): return String(describing: kw)
        case let .opcode(op): return String(describing: op)
        case let .identifier(kind, id):
            let kindDesc: String
            switch kind {
            case .basicBlock: kindDesc = "'"
            case .global: kindDesc = "@"
            case .key: kindDesc = "#"
            case .temporary: kindDesc = "%"
            case .type: kindDesc = "$"
            }
            return kindDesc + id
        case let .stringLiteral(str): return "\"\(str)\""
        case .newLine: return "a new line"
        case .indent: return "an indentation"
        case let .anonymousIdentifier(b, i):
            return "%\(b).\(i)"
        case let .attribute(attr):
            return "!" + String(describing: attr)
        }
    }
}
