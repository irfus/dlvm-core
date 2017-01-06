//
//  Parse.swift
//  DLVM
//
//  Created by Richard Wei on 12/22/16.
//
//

import Parsey

/// Local primitive parsers
fileprivate let identifier = Lexer.regex("[a-zA-Z_][a-zA-Z0-9_]*")
fileprivate let number = Lexer.unsignedInteger ^^ { Int($0)! }
fileprivate let lineComments = ("//" ~~> Lexer.string(until: "\n") <~~ Lexer.newLine)+
fileprivate let spaces = (Lexer.whitespace | Lexer.tab)+
fileprivate let newLines = Lexer.newLine+
fileprivate let linebreaks = (newLines | lineComments).amid(spaces.?)+ .. "a linebreak"

protocol Parsible {
    static var parser: Parser<Self> { get }
}

