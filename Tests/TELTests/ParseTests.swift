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
            try Expression.parser.parse("[v0, W     v]")
            try Expression.parser.parse("[v0, -v0+v1*v4]")
            try Expression.parser.parse("[v0, -(v0+v1)*v4]")
        }
        catch {
            XCTFail("\(error)")
        }
    }

    func testParseStatement() {
        do {
            try Statement.parser.parse("h: hidden[2x1] = tanh(W x + b)")
        }
        catch {
            XCTFail("\(error)")
        }
    }
    
}
