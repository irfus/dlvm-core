//
//  ParseTests.swift
//  DLVM
//
//  Created by Richard Wei on 5/30/17.
//
//

import XCTest
@testable import DLParse

class ParseTests: XCTestCase {
    func testType() throws {
        let types = [
            "f32",
            "[1 x f32]",
            "<4 x 3 x 10 x i64>",
            "[1 x <4 x 3 x i8>]",
            "(i8, <4 x 3 x bool>)"
        ]
        for type in types {
            do {
                let parser = try Parser(text: type)
                _ = try parser.parseType()
            } catch {
                XCTFail(String(describing: error) + " when parsing " + type)
            }
        }
    }

    func testUse() throws {
        let uses = [
            "%x : f32",
            "@func : [1 x f32]",
            "(@vec1 : <4 x 3 x 10 x i64>, @vec2 : <3 x f32>) : (<4 x 3 x 10 x i64>, <3 x f32>)",
            "@val : [1 x <4 x 3 x i8>]",
            "%tup : (i8, <4 x 3 x bool>)",
            "false :///comments\n\n <4 x 3 x bool>",
            "0 : <4 x 3 x i32>",
        ]
        for type in uses {
            do {
                let parser = try Parser(text: type)
                _ = try parser.parseUse()
            } catch {
                XCTFail(String(describing: error) + " when parsing " + type)
            }
        }
    }
    
    static var allTests : [(String, (ParseTests) -> () throws -> Void)] {
        return [
        ]
    }
}
