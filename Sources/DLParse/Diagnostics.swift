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
}

public enum ParseError : Error {
    case unexpectedIdentifierKind(IdentifierKind, Token)
    case unexpectedEndOfInput(expected: String)
    case unexpectedToken(expected: String, Token)
    case noDimensionsInTensorShape(Token)
}

public extension ParseError {
    /// Location of the error, nil if EOF
    var location: SourceLocation? {
        switch self {
        case let .unexpectedIdentifierKind(_, tok):
            return tok.range.lowerBound
        case .unexpectedEndOfInput(_):
            return nil
        case let .unexpectedToken(_, tok):
            return tok.startLocation
        case let .noDimensionsInTensorShape(tok):
            return tok.startLocation
        }
    }
}

extension ParseError : CustomStringConvertible {
    public var description : String {
        switch self {
        case let .unexpectedIdentifierKind(kind, tok):
            return "Unexpected kind of identifier (\(kind)) at \(tok.startLocation)"
        case let .unexpectedEndOfInput(expected: expected):
            return "Expected \(expected) but reached the end of input"
        case let .unexpectedToken(expected: expected, tok):
            return "Expected \(expected) but found the token at \(tok.startLocation)"
        case let .noDimensionsInTensorShape(tok):
            return "No dimensions in tensor type at \(tok.startLocation). If you'd like it to be a scalar, use the data type (e.g. f32) directly."
        }
    }
}

extension LexicalError : CustomStringConvertible {
    public var description: String {
        switch self {
        case let .illegalIdentifier(range):
            return "Illegal identifier at \(range)"
        case let .illegalNumber(range):
            return "Illegal number at \(range)"
        case let .unexpectedToken(loc):
            return "Unexpected token at \(loc)"
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
        case .gradient: return "gradient"
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
        case .newLine: return "a new line"
        case let .keyword(kw): return String(describing: kw)
        case let .opcode(op): return String(describing: op)
        case let .identifier(kind, id):
            return String(describing: kind) + "identifier \"\(id)\""
        }
    }
}
