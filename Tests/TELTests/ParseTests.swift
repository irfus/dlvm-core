//
//  ParseTests.swift
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

import XCTest
@testable import TEL
@testable import Parsey

class ParseTests : XCTestCase {

    func testParseExpression() {
        do {
            _ = try Expression.parser.parse("[v0, W •    v]")
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
            _ = try Declaration.parser.parse("h: hidden[2x1] = tanh(W . x + b)")
            _ = try Declaration.parser.parse("recurrent t {\n h: hidden[2x1] = tanh(W . x + b) \n}")
        }
        catch {
            XCTFail("\(error)")
        }
    }

    func testParseMacro() {
        do {
            _ = try Attribute.parser.parse("type float8")
            _ = try Attribute.parser.parse("type float16")
            _ = try Attribute.parser.parse("type int32")
            _ = try Attribute.parser.parse("type int64")
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
