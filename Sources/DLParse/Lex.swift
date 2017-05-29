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
    case stage
    case `struct`, `func`, `var`, `let`
    case at, to, from, by
    case then, `else`
    case broadcast
    case wrt, keeping
    case left, right
    case void
    case zero, undefined, null
}

public enum Punctuation {
    case leftParenthesis, rightParenthesis
    case leftSquareBracket, rightSquareBracket
    case leftAngleBracket, rightAngleBracket
    case leftCurlyBracket, rightCurlyBracket
    case colon
    case equal
    case pound
    case rightArrow
    case comma
    case times
    case star
}

public enum IdentifierKind {
    case attribute
    case basicBlock
    case temporary
    case global
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

public extension TokenKind {
    func makeToken(in range: SourceRange) -> Token {
        return Token(kind: self, range: range)
    }
}

struct LexStream {
    var characters: String.UnicodeScalarView
    var location: SourceLocation
}

extension String.UnicodeScalarView {
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

import class Foundation.NSRegularExpression
import struct Foundation.NSRange

extension LexStream {
    init(_ text: String) {
        characters = text.unicodeScalars
        location = SourceLocation()
    }
    
    mutating func consume(_ n: Int) {
        characters.removeFirst(n)
        location.advance(by: n)
    }
}

extension UnicodeScalar {
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

extension LexStream {
    private mutating func scanPunctuation() throws -> Token {
        let loc = location
        let tok: Token
        switch characters[characters.startIndex] {
        case "(": tok = Token(kind: .punctuation(.leftParenthesis), range: loc..<loc+1)
        case ")": tok = Token(kind: .punctuation(.rightParenthesis), range: loc..<loc+1)
        case "[": tok = Token(kind: .punctuation(.leftSquareBracket), range: loc..<loc+1)
        case "]": tok = Token(kind: .punctuation(.rightSquareBracket), range: loc..<loc+1)
        case "<": tok = Token(kind: .punctuation(.leftAngleBracket), range: loc..<loc+1)
        case ">": tok = Token(kind: .punctuation(.rightAngleBracket), range: loc..<loc+1)
        case "{": tok = Token(kind: .punctuation(.leftCurlyBracket), range: loc..<loc+1)
        case "}": tok = Token(kind: .punctuation(.rightCurlyBracket), range: loc..<loc+1)
        case ":": tok = Token(kind: .punctuation(.colon), range: loc..<loc+1)
        case "=": tok = Token(kind: .punctuation(.equal), range: loc..<loc+1)
        case ",": tok = Token(kind: .punctuation(.comma), range: loc..<loc+1)
        case "#": tok = Token(kind: .punctuation(.pound), range: loc..<loc+1)
        case "x": tok = Token(kind: .punctuation(.times), range: loc..<loc+1)
        case "*": tok = Token(kind: .punctuation(.star), range: loc..<loc+1)
        case "!": consume(1); return try lexIdentifier(ofKind: .attribute)
        case "@": consume(1); return try lexIdentifier(ofKind: .global)
        case "%": consume(1); return try lexIdentifier(ofKind: .temporary)
        case "'": consume(1); return try lexIdentifier(ofKind: .basicBlock)
        case "-":
            guard characters.dropFirst().first == ">" else {
                throw TokenError.illegalToken(loc.advanced(by: 1))
            }
            consume(1)
            tok = Token(kind: .punctuation(.rightArrow), range: loc..<loc+2)
        default:
            throw TokenError.illegalToken(loc)
        }
        consume(1)
        return tok
    }

    private mutating func scanNumber() throws -> Token {
        let endOfWhole = characters.index(where: { !$0.isNumber }) ?? characters.endIndex
        var number = characters.prefix(upTo: endOfWhole)
        let startLoc = location
        consume(number.count)
        /// If there's a dot, lex float literal
        if endOfWhole < characters.endIndex, characters[endOfWhole] == "." {
            consume(1)
            number.append(".")
            /// Has decimal dot
            let afterDot = characters.index(after: endOfWhole)
            guard afterDot < characters.endIndex, characters[afterDot].isNumber else {
                throw TokenError.illegalNumber(startLoc..<location)
            }
            let decimal = characters.prefix(while: { $0.isNumber })
            consume(decimal.count)
            number.append(contentsOf: decimal)
            guard let float = FloatLiteralType(String(number)) else {
                throw TokenError.illegalNumber(startLoc..<location)
            }
            return Token(kind: .float(float), range: startLoc..<location)
        }
        /// Integer literal
        guard let integer = Int(String(number)) else {
            throw TokenError.illegalNumber(startLoc..<location)
        }
        return Token(kind: .integer(integer), range: location..<location+characters.count)
    }

    private mutating func scanLetter() throws -> Token {
        let prefix = characters.prefix(while: {
            !($0.isWhitespace || $0.isNewLine || $0.isPunctuation)
        })
        let startLoc = location
        consume(prefix.count)
        let kind: TokenKind
        switch prefix {
        /// Keywords
        case "module": kind = .keyword(.module)
        case "stage": kind = .keyword(.stage)
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
                throw TokenError.illegalToken(startLoc)
            }
            kind = .dataType(.int(UInt(size)))
        default:
            throw TokenError.illegalToken(startLoc)
        }
        return Token(kind: kind, range: startLoc..<location)
    }

    mutating func lexIdentifier(ofKind kind: IdentifierKind) throws -> Token {
        let prefix = characters.prefix(while: {
            !($0.isWhitespace || $0.isNewLine || $0.isPunctuation)
        })
        let startLoc = location
        guard prefix.matchesRegex(identifierPattern) else {
            throw TokenError.illegalToken(location)
        }
        consume(prefix.count)
        return Token(kind: .identifier(kind, String(prefix)), range: startLoc..<location)
    }

    mutating func lex() throws -> [Token] {
        var tokens: [Token] = []
        while let first = characters.first {
            let tok: Token
            if first.isPunctuation {
                tok = try scanPunctuation()
            }
            else if first.isNumber {
                tok = try scanNumber()
            }
            else if first.isAlphabet {
                tok = try scanLetter()
            }
            else if first.isNewLine {
                characters.removeFirst()
                tok = Token(kind: .newLine, range: location..<location)
                location.advanceToNewLine()
            }
            /// Ignore whitespaces
            else if first.isWhitespace {
                consume(1)
                continue
            }
            /// Ignore comments
            else if characters.starts(with: "//") {
                let comment = characters.prefix(while: { !$0.isNewLine })
                consume(comment.count)
                continue
            }
            else {
                throw TokenError.illegalToken(location)
            }
            tokens.append(tok)
        }
        return tokens
    }
}
