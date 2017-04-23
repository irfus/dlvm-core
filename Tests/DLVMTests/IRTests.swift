//
//  IRTests.swift
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
@testable import DLVM

class IRTests: XCTestCase {

    let builder = IRBuilder(moduleName: "IRTest")

    func testWriteGlobal() {
        let val1 = builder.buildGlobalValue(named: "one", kind: .constant,
                                            type: .scalar(.int(32)),
                                            initializer: .constant(.binary(
                                                .associative(.arithmetic(.add)),
                                                LiteralValue(type: .scalar(.int(32)),
                                                             literal: .scalar(.int(10))).makeUse(),
                                                LiteralValue(type: .scalar(.int(32)),
                                                             literal: .scalar(.int(20))).makeUse())))
        XCTAssertEqual("\(val1.value)", "let @one : i32 = (add 10 : i32, 20 : i32) : i32")
        let val2 = builder.buildGlobalValue(named: "two", kind: .constant,
                                            type: Type.scalar(.int(32)).pointer,
                                            initializer: val1)
        XCTAssertEqual("\(val2.value)", "let @two : *i32 = @one : *i32")
    }

    func testWriteStruct() {
        let struct1 = builder.buildStruct(named: "TestStruct1", fields: [
            "foo" : .scalar(.int(32)),
            "bar" : .tensor([1, 3, 4], .float(.double)),
            "baz" : .array(.array(.box(.tensor([3], .int(32)), .normal), 3), 4)
        ], attributes: [ .packed ])
        XCTAssertEqual(struct1.description,
                       "!packed\nstruct $TestStruct1 {\n    foo: i32\n    bar: <1x3x4.f64>\n    baz: [4 x [3 x box{<3.i32>}]]\n}")
        let structLit = builder.makeLiteral(LiteralValue(type: .struct(struct1), literal: .struct([
            ("foo", .literal(LiteralValue(type: .scalar(.int(32)), literal: .undefined))),
            ("bar", .literal(LiteralValue(type: .tensor([1, 3, 4], .float(.double)), literal: .undefined))),
            ("baz", .literal(LiteralValue(type: .array(.array(.box(.tensor([3], .int(32)), .normal), 3), 4), literal: .undefined)))
        ])))
        let structGlobal = builder.buildGlobalValue(named: "struct1.value",
                                                    kind: .variable,
                                                    type: .struct(struct1),
                                                    initializer: structLit)
        XCTAssertEqual("\(structGlobal.value)", "var @struct1.value : $TestStruct1 = {#foo = undefined : i32, #bar = undefined : <1x3x4.f64>, #baz = undefined : [4 x [3 x box{<3.i32>}]]} : $TestStruct1")
    }

    static var allTests : [(String, (IRTests) -> () throws -> Void)] {
        return [
            ("testWriteGlobal", testWriteGlobal),
            ("testWriteStruct", testWriteStruct),
        ]
    }
    
}
