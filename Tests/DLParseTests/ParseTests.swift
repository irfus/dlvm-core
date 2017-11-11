//
//  ParseTests.swift
//  DLVM
//
//  Copyright 2016-2017 The DLVM Team.
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

class ParseTests: XCTestCase {
    func testType() throws {
        let types = [
            "f32",
            "[1 x f32]",
            "*<4 x 3 x 10 x i64>",
            "[1 x <4 x 3 x i8>]",
            "(i8, <4 x 3 x bool>)"
        ]
        for type in types {
            do {
                let parser = try Parser(text: type)
                _ = try parser.parseType()
            } catch {
                XCTFail(String(describing: error) + " when parsing \"\(type)\"")
            }
        }
    }

    func testUse() throws {
        let uses = [
            "2 : f32",
            "1 : [1 x f32]",
            """
            (3 : <4 x 3 x 10 x i64>, <1.0: f32, 2.0: f32, 3.0: f32> : <3 x f32>)\
            : (<4 x 3 x 10 x i64>, <3 x f32>)
            """,
            "100.0 : [1 x <4 x 3 x i8>]",
            "100.0 : (i8, <4 x 3 x bool>)",
            "false :///comments\n\n <4 x 3 x bool>",
            "0 : <4 x 3 x i32>",
            "{ #hello = false : bool, #value = 100.0: f32 } : f32"
        ]
        for type in uses {
            do {
                let parser = try Parser(text: type)
                _ = try parser.parseUse(in: nil)
            } catch {
                XCTFail(String(describing: error) + " when parsing \"\(type)\"")
            }
        }
    }

    func testInstructionKind() throws {
        let uses = [
            "add 1: f32, <2: f32, 3: f32>: <2 x f32>",
            "elementPointer 1: *f32 at #name1, #name2, 3, 4",
        ]
        for type in uses {
            do {
                let parser = try Parser(text: type)
                _ = try parser.parseInstructionKind(in: nil)
            } catch {
                XCTFail(String(describing: error) + " when parsing \"\(type)\"")
            }
        }
    }
    
    static var allTests : [(String, (ParseTests) -> () throws -> Void)] {
        return [
        ]
    }
}
