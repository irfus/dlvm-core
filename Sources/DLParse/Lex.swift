//
//  Lex.swift
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

import enum DLVM.InstructionKind
import enum DLVM.DataType

public enum Keyword {
    case module
    case stage, raw, canonical
    case `struct`, `func`, `var`, `let`
    case at, to, from, by
    case then, `else`
    case broadcast
    case wrt, keeping
    case left, right
    case void
    case zero, undefined, null
    case `true`, `false`
    case scalar
}

public enum Punctuation {
    case leftParenthesis, rightParenthesis
    case leftSquareBracket, rightSquareBracket
    case leftAngleBracket, rightAngleBracket
    case leftCurlyBracket, rightCurlyBracket
    case colon
    case equal
    case rightArrow
    case comma
    case times
    case star
}

public enum IdentifierKind {
    case attribute
    case basicBlock
    case temporary
    case type
    case global
    case key
}

public enum TokenKind {
    case punctuation(Punctuation)
    case keyword(Keyword)
    case opcode(InstructionKind.Opcode)
    case integer(IntegerLiteralType)
    case float(FloatLiteralType)
    case identifier(IdentifierKind, String)
    case dataType(DataType)
    case newLine
}

public extension TokenKind {
    func isIdentifier(ofKind kind: IdentifierKind) -> Bool {
        guard case .identifier(kind, _) = self else { return false }
        return true
    }

    var isInteger: Bool {
        guard case .integer(_) = self else { return false }
        return true
    }

    var isFloat: Bool {
        guard case .float(_) = self else { return false }
        return true
    }
}

extension TokenKind : Equatable {
    public static func == (lhs: TokenKind, rhs: TokenKind) -> Bool {
        switch (lhs, rhs) {
        case let (.punctuation(p1), .punctuation(p2)):
            return p1 == p2
        case let (.keyword(k1), .keyword(k2)):
            return k1 == k2
        case let (.opcode(o1), .opcode(o2)):
            return o1 == o2
        case let (.integer(i1), .integer(i2)):
            return i1 == i2
        case let (.float(f1), .float(f2)):
            return f1 == f2
        case let (.identifier(k1, i1), .identifier(k2, i2)):
            return k1 == k2 && i1 == i2
        case let (.dataType(d1), .dataType(d2)):
            return d1 == d2
        case (.newLine, .newLine):
            return true
        default:
            return false
        }
    }
}

public struct Token {
    public let kind: TokenKind
    public let range: SourceRange
}

extension TokenKind {
    func makeToken(in range: SourceRange) -> Token {
        return Token(kind: self, range: range)
    }
}

public extension Token {
    var startLocation: SourceLocation {
        return range.lowerBound
    }

    var endLocation: SourceLocation {
        return range.upperBound
    }
}

public class Lexer {
    public fileprivate(set) var characters: String.UnicodeScalarView
    public fileprivate(set) var location = SourceLocation()

    public init(text: String) {
        characters = text.unicodeScalars
    }
}

import class Foundation.NSRegularExpression
import struct Foundation.NSRange

private extension String.UnicodeScalarView {
    static func ~= (pattern: String, value: String.UnicodeScalarView) -> Bool {
        return pattern.unicodeScalars.elementsEqual(value)
    }

    func starts(with possiblePrefix: String) -> Bool {
        return starts(with: possiblePrefix.unicodeScalars)
    }

    func matchesRegex(_ regex: NSRegularExpression) -> Bool {
        let matches = regex.matches(in: String(self),
                                    options: [ .anchored ],
                                    range: NSRange(location: 0, length: count))
        return matches.count == 1 && matches[0].range.length == count
    }
}

private extension UnicodeScalar {
    var isNewLine: Bool {
        switch self {
        case "\n", "\r": return true
        default: return false
        }
    }

    var isWhitespace: Bool {
        switch self {
        case " ", "\t": return true
        default: return false
        }
    }

    var isPunctuation: Bool {
        return (33...45).contains(value)
            || (58...64).contains(value)
            || (91...93).contains(value)
            || (123...126).contains(value)
    }

    var isNumber: Bool {
        return (48...57).contains(value)
    }

    var isAlphabet: Bool {
        return (65...90).contains(value)
            || (97...122).contains(value)
    }
}

private let identifierPattern = try! NSRegularExpression(pattern: "[a-zA-Z0-9_][a-zA-Z0-9_.]*",
                                                         options: [ .dotMatchesLineSeparators ])

private extension Lexer {
    
    func advance(by n: Int) {
        characters.removeFirst(n)
        location.advance(by: n)
    }

    func advanceToNewLine() {
        characters.removeFirst()
        location.advanceToNewLine()
    }

    func lexIdentifier(ofKind kind: IdentifierKind) throws -> Token {
        let prefix = characters.prefix(while: {
            !($0.isWhitespace || $0.isNewLine || $0.isPunctuation)
        })
        let startLoc = location
        advance(by: prefix.count)
        guard prefix.matchesRegex(identifierPattern) else {
            throw LexicalError.illegalIdentifier(startLoc..<location)
        }
        return Token(kind: .identifier(kind, String(prefix)), range: startLoc..<location)
    }

    func scanPunctuation() throws -> Token {
        let startLoc = location
        let kind: TokenKind
        let first = characters[characters.startIndex]
        advance(by: 1)
        var count = 1
        switch first {
        case "(": kind = .punctuation(.leftParenthesis)
        case ")": kind = .punctuation(.rightParenthesis)
        case "[": kind = .punctuation(.leftSquareBracket)
        case "]": kind = .punctuation(.rightSquareBracket)
        case "<": kind = .punctuation(.leftAngleBracket)
        case ">": kind = .punctuation(.rightAngleBracket)
        case "{": kind = .punctuation(.leftCurlyBracket)
        case "}": kind = .punctuation(.rightCurlyBracket)
        case ":": kind = .punctuation(.colon)
        case "=": kind = .punctuation(.equal)
        case ",": kind = .punctuation(.comma)
        case "x": kind = .punctuation(.times)
        case "*": kind = .punctuation(.star)
        case "#": return try lexIdentifier(ofKind: .key)
        case "!": return try lexIdentifier(ofKind: .attribute)
        case "@": return try lexIdentifier(ofKind: .global)
        case "%": return try lexIdentifier(ofKind: .temporary)
        case "$": return try lexIdentifier(ofKind: .type)
        case "'": return try lexIdentifier(ofKind: .basicBlock)
        case "-":
            guard characters.first == ">" else {
                throw LexicalError.unexpectedToken(location)
            }
            advance(by: 1)
            count += 1
            kind = .punctuation(.rightArrow)
        default:
            throw LexicalError.unexpectedToken(startLoc)
        }
        return Token(kind: kind, range: startLoc..<startLoc.advanced(by: count))
    }

    func scanNumber() throws -> Token {
        let endOfWhole = characters.index(where: { !$0.isNumber }) ?? characters.endIndex
        var number = characters.prefix(upTo: endOfWhole)
        let startLoc = location
        advance(by: number.count)
        /// If there's a dot, lex float literal
        if endOfWhole < characters.endIndex, characters[endOfWhole] == "." {
            advance(by: 1)
            number.append(".")
            /// Has decimal dot
            let afterDot = characters.index(after: endOfWhole)
            guard afterDot < characters.endIndex, characters[afterDot].isNumber else {
                throw LexicalError.illegalNumber(startLoc..<location)
            }
            let decimal = characters.prefix(while: { $0.isNumber })
            advance(by: decimal.count)
            number.append(contentsOf: decimal)
            guard let float = FloatLiteralType(String(number)) else {
                throw LexicalError.illegalNumber(startLoc..<location)
            }
            return Token(kind: .float(float), range: startLoc..<location)
        }
        /// Integer literal
        guard let integer = Int(String(number)) else {
            throw LexicalError.illegalNumber(startLoc..<location)
        }
        return Token(kind: .integer(integer), range: location..<location+characters.count)
    }

    func scanLetter() throws -> Token {
        let prefix = characters.prefix(while: {
            !($0.isWhitespace || $0.isNewLine || $0.isPunctuation)
        })
        let startLoc = location
        advance(by: prefix.count)
        let kind: TokenKind
        switch prefix {
        /// Keywords
        case "module": kind = .keyword(.module)
        case "stage": kind = .keyword(.stage)
        case "raw": kind = .keyword(.raw)
        case "canonical": kind = .keyword(.canonical)
        case "func": kind = .keyword(.func)
        case "struct": kind = .keyword(.struct)
        case "var": kind = .keyword(.var)
        case "let": kind = .keyword(.let)
        case "at": kind = .keyword(.at)
        case "to": kind = .keyword(.to)
        case "from": kind = .keyword(.from)
        case "by": kind = .keyword(.by)
        case "then": kind = .keyword(.then)
        case "else": kind = .keyword(.else)
        case "broadcast": kind = .keyword(.broadcast)
        case "wrt": kind = .keyword(.wrt)
        case "keeping": kind = .keyword(.keeping)
        case "left": kind = .keyword(.left)
        case "right": kind = .keyword(.right)
        case "void": kind = .keyword(.void)
        case "zero": kind = .keyword(.zero)
        case "undefined": kind = .keyword(.undefined)
        case "null": kind = .keyword(.null)
        case "true": kind = .keyword(.true)
        case "false": kind = .keyword(.false)
        case "scalar": kind = .keyword(.scalar)
        /// Opcode
        case "branch": kind = .opcode(.branch)
        case "conditional": kind = .opcode(.conditional)
        case "return": kind = .opcode(.return)
        case "dataTypeCast": kind = .opcode(.dataTypeCast)
        case "scan": kind = .opcode(.scan)
        case "reduce": kind = .opcode(.reduce)
        case "matrixMultiply": kind = .opcode(.matrixMultiply)
        case "concatenate": kind = .opcode(.concatenate)
        case "transpose": kind = .opcode(.transpose)
        case "shapeCast": kind = .opcode(.shapeCast)
        case "bitCast": kind = .opcode(.bitCast)
        case "extract": kind = .opcode(.extract)
        case "insert": kind = .opcode(.insert)
        case "apply": kind = .opcode(.apply)
        case "gradient": kind = .opcode(.gradient)
        case "allocateStack": kind = .opcode(.allocateStack)
        case "allocateHeap": kind = .opcode(.allocateHeap)
        case "allocateBox": kind = .opcode(.allocateBox)
        case "projectBox": kind = .opcode(.projectBox)
        case "retain": kind = .opcode(.retain)
        case "release": kind = .opcode(.release)
        case "deallocate": kind = .opcode(.deallocate)
        case "load": kind = .opcode(.load)
        case "store": kind = .opcode(.store)
        case "elementPointer": kind = .opcode(.elementPointer)
        case "copy": kind = .opcode(.copy)
        case "trap": kind = .opcode(.trap)
        case "lessThan": kind = .opcode(.binaryOp(.comparison(.lessThan)))
        case "lessThanOrEqual": kind = .opcode(.binaryOp(.comparison(.lessThanOrEqual)))
        case "greaterThan": kind = .opcode(.binaryOp(.comparison(.lessThanOrEqual)))
        case "greaterThanOrEqual": kind = .opcode(.binaryOp(.comparison(.greaterThanOrEqual)))
        case "equal": kind = .opcode(.binaryOp(.comparison(.equal)))
        case "notEqual": kind = .opcode(.binaryOp(.comparison(.notEqual)))
        case "and": kind = .opcode(.binaryOp(.associative(.and)))
        case "or": kind = .opcode(.binaryOp(.associative(.or)))
        case "add": kind = .opcode(.binaryOp(.associative(.add)))
        case "subtract": kind = .opcode(.binaryOp(.associative(.subtract)))
        case "multiply": kind = .opcode(.binaryOp(.associative(.multiply)))
        case "divide": kind = .opcode(.binaryOp(.associative(.divide)))
        case "min": kind = .opcode(.binaryOp(.associative(.min)))
        case "max": kind = .opcode(.binaryOp(.associative(.max)))
        case "truncateDivide": kind = .opcode(.binaryOp(.associative(.truncateDivide)))
        case "floorDivide": kind = .opcode(.binaryOp(.associative(.floorDivide)))
        case "modulo": kind = .opcode(.binaryOp(.associative(.modulo)))
        case "power": kind = .opcode(.binaryOp(.associative(.power)))
        case "mean": kind = .opcode(.binaryOp(.associative(.mean)))
        case "tanh": kind = .opcode(.unaryOp(.tanh))
        case "log": kind = .opcode(.unaryOp(.log))
        case "exp": kind = .opcode(.unaryOp(.exp))
        case "negate": kind = .opcode(.unaryOp(.negate))
        case "sign": kind = .opcode(.unaryOp(.sign))
        case "square": kind = .opcode(.unaryOp(.square))
        case "sign": kind = .opcode(.unaryOp(.sign))
        case "square": kind = .opcode(.unaryOp(.square))
        case "sqrt": kind = .opcode(.unaryOp(.sqrt))
        case "round": kind = .opcode(.unaryOp(.round))
        case "rsqrt": kind = .opcode(.unaryOp(.rsqrt))
        case "ceil": kind = .opcode(.unaryOp(.ceil))
        case "floor": kind = .opcode(.unaryOp(.floor))
        case "tan": kind = .opcode(.unaryOp(.tan))
        case "cos": kind = .opcode(.unaryOp(.cos))
        case "sin": kind = .opcode(.unaryOp(.sin))
        case "acos": kind = .opcode(.unaryOp(.acos))
        case "asin": kind = .opcode(.unaryOp(.asin))
        case "atan": kind = .opcode(.unaryOp(.atan))
        case "lgamma": kind = .opcode(.unaryOp(.lgamma))
        case "digamma": kind = .opcode(.unaryOp(.digamma))
        case "erf": kind = .opcode(.unaryOp(.erf))
        case "erfc": kind = .opcode(.unaryOp(.erfc))
        case "rint": kind = .opcode(.unaryOp(.rint))
        case "not": kind = .opcode(.unaryOp(.not))
        case "x": kind = .punctuation(.times)
        case "f16": kind = .dataType(.float(.half))
        case "f32": kind = .dataType(.float(.single))
        case "f64": kind = .dataType(.float(.double))
        case "bool": kind = .dataType(.bool)
        case _ where prefix.first == "i":
            let rest = prefix.dropFirst()
            guard rest.forAll({$0.isNumber}), let size = Int(String(rest)) else {
                throw LexicalError.illegalNumber(startLoc+1..<location)
            }
            kind = .dataType(.int(UInt(size)))
        default:
            throw LexicalError.unexpectedToken(startLoc)
        }
        return Token(kind: kind, range: startLoc..<location)
    }

}

public extension Lexer {
    func performLexing() throws -> [Token] {
        var tokens: [Token] = []
        while let first = characters.first {
            let tok: Token
            /// Parse tokens starting with a punctuation
            if first.isPunctuation {
                tok = try scanPunctuation()
            }
            /// Parse tokens starting with a number
            else if first.isNumber {
                tok = try scanNumber()
            }
            /// Parse tokens starting with a letter
            else if first.isAlphabet {
                tok = try scanLetter()
            }
            /// Parse new line
            else if first.isNewLine {
                tok = Token(kind: .newLine, range: location..<location)
                advanceToNewLine()
            }
            /// Ignore whitespace
            else if first.isWhitespace {
                advance(by: 1)
                continue
            }
            /// Ignore line comment
            else if characters.starts(with: "//") {
                let comment = characters.prefix(while: { !$0.isNewLine })
                advance(by: comment.count)
                continue
            }
            /// Illegal start character
            else {
                throw LexicalError.unexpectedToken(location)
            }
            tokens.append(tok)
        }
        return tokens
    }
}
