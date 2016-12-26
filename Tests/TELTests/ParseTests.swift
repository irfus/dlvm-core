//
//  ParseTests.swift
//  DLVM
//
//  Created by Richard Wei on 12/21/16.
//
//

import XCTest
@testable import TEL
@testable import Parsey

class ParseTests : XCTestCase {

    func testParseExpression() {
        do {
            _ = try Expression.parser.parse("[v0, W     v]")
            _ = try Expression.parser.parse("[v0, -v0+v1*v4]")
            _ = try Expression.parser.parse("[v0, -(v0+v1)*v4]")
            _ = try Expression.parser.parse("0")
            _ = try Expression.parser.parse("0.0")
        }
        catch {
            XCTFail("\(error)")
        }
    }

    func testParseDeclaration() {
        do {
            _ = try Declaration.parser.parse("i: in[2x1]")
            _ = try Declaration.parser.parse("h: hidden[2x1] = tanh(W x + b)")
            _ = try Declaration.parser.parse("recurrent t {\n h: hidden[2x1] = tanh(W x + b) \n}")
        }
        catch {
            XCTFail("\(error)")
        }
    }

    func testParseMacro() {
        do {
            _ = try Macro.parser.parse("#type float8")
            _ = try Macro.parser.parse("#type float16")
            _ = try Macro.parser.parse("#type int32")
            _ = try Macro.parser.parse("#type int64")
        }
        catch {
            XCTFail("\(error)")
        }
    }

    public static var allTests: [(String, (ParseTests) -> () throws -> Void)] {
        return [
            ("testParseExpression", testParseExpression),
            ("testParseDeclaration", testParseDeclaration),
            ("testParseMacro", testParseMacro),
        ]
    }

}
