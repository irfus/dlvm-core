//
//  LexTests.swift
//  DLVM
//
//  Copyright 2016-2018 The DLVM Team.
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
@testable import DLParse

class LexTests : XCTestCase {
    func testBasicLexing() throws {
        do {
            let code = """
               func @mnist.inference.impl.gradient: (<1 x 784 x f32>)\
               // comments\n\n
               """
            let lexer = Lexer(text: code)
            let tokens = try lexer.performLexing().map{$0.kind}
            let expectedTokens: [TokenKind] = [
                .keyword(.func),
                .identifier(.global, "mnist.inference.impl.gradient"),
                .punctuation(.colon),
                .punctuation(.leftParenthesis),
                .punctuation(.leftAngleBracket),
                .integer(1),
                .punctuation(.times),
                .integer(784),
                .punctuation(.times),
                .dataType(.float(.single)),
                .punctuation(.rightAngleBracket),
                .punctuation(.rightParenthesis),
                .newLine,
                .newLine
            ]
            XCTAssertEqual(tokens, expectedTokens)
        } catch {
            XCTFail(String(describing: error))
        }
    }

    func testStringLiteralLexing() throws {
        do {
            let code = "\"dasdasa\"\n\n\"das\\n\\t\\rdasa\""
            let lexer = Lexer(text: code)
            _ = try lexer.performLexing()
        } catch {
            XCTFail(String(describing: error))
        }
    }

    func testStructLexing() throws {
        do {
            let code = """
                struct $TestStruct1 {
                    #foo: i32,
                    #bar: <1 x 3 x 4 x f64>,
                    #baz: [4 x [3 x <3 x i32>]]
                }
                """
            let lexer = Lexer(text: code)
            _ = try lexer.performLexing()
        } catch {
            XCTFail(String(describing: error))
        }
    }

    func testFunctionLexing() throws {
        do {
            let code = """
                func @bar: (f32, f32) -> i32 {
                'entry(%x: f32, %y: f32):
                    %0.0 = multiply 5: i32, 8: i32
                    %0.1 = equal %0.0: i32, 1: i32
                    conditional %0.1: bool then 'then(0: i32) else 'else(1: i32)
                'then(%x: i32):
                    branch 'cont(%x: i32)
                'else(%x: i32):
                    branch 'cont(%x: i32)
                'cont(%x: i32):
                    return %x: i32\n
                """
            let lexer = Lexer(text: code)
            _ = try lexer.performLexing()
        } catch {
            XCTFail(String(describing: error))
        }
    }

    static var allTests: [(String, (LexTests) -> () throws -> Void)] {
        return [
            ("testBasicLexing", testBasicLexing),
            ("testStringLiteralLexing", testStringLiteralLexing),
            ("testStructLexing", testStructLexing),
            ("testFunctionLexing", testFunctionLexing)
        ]
    }
}
