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
            try Expression.negateParser.parse("-v")
        }
        catch {
            XCTFail("\(error)")
        }
    }
    
}
