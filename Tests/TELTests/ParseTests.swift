//
//  ParseTests.swift
//  DLVM
//
//  Created by Richard Wei on 12/21/16.
//
//

import XCTest
@testable import TEL

class ParseTests : XCTestCase {

    func testParseExpression() {
        do {
            _ = try Expression.parser.parse("[v0, W     v]")
            _ = try Expression.parser.parse("[v0, -v0+v1*v4]")
            _ = try Expression.parser.parse("[v0, -(v0+v1)*v4]")
        }
        catch {
            XCTFail("\(error)")
        }
    }

    func testParseStatement() {
        do {
            _ = try Declaration.parser.parse("h: hidden[2x1] = tanh(W x + b)")
            _ = try Declaration.parser.parse("recurrent t {\n h: hidden[2x1] = tanh(W x + b) \n}")
        }
        catch {
            XCTFail("\(error)")
        }
    }
    
}
