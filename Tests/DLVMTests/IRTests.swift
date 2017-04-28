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
                                            type: .int(32),
                                            initializer: .constant(.binary(
                                                .associative(.arithmetic(.add)),
                                                .literal(.int(32), .scalar(.int(10))),
                                                .literal(.int(32), .scalar(.int(20))), nil)))
        XCTAssertEqual("\(val1.value)", "let @one : i32 = (add 10 : i32, 20 : i32) : i32")
        let val2 = builder.buildGlobalValue(named: "two", kind: .constant,
                                            type: Type.int(32).pointer,
                                            initializer: val1)
        XCTAssertEqual("\(val2.value)", "let @two : *i32 = @one : *i32")
    }

    func testWriteStruct() {
        let struct1 = builder.buildStruct(named: "TestStruct1", fields: [
            "foo" : .int(32),
            "bar" : .tensor([1, 3, 4], .float(.double)),
            "baz" : .array(.array(.tensor([3], .int(32)), 3), 4)
        ], attributes: [ .packed ])
        XCTAssertEqual(struct1.description,
                       "!packed\nstruct $TestStruct1 {\n    foo: i32\n    bar: <1x3x4.f64>\n    baz: [4 x [3 x <3.i32>]]\n}")
        let structLit = builder.makeLiteral(.struct([
            ("foo", .literal(.int(32), 100000)),
            ("bar", .literal(.tensor([1, 3, 4], .float(.double)), .undefined)),
            ("baz", .literal(.array(.array(.tensor([3], .int(32)), 3), 4), .undefined))
        ]), ofType: .struct(struct1))
        let structGlobal = builder.buildGlobalValue(named: "struct1.value",
                                                    kind: .variable,
                                                    type: .struct(struct1),
                                                    initializer: structLit)
        XCTAssertEqual("\(structGlobal.value)", "var @struct1.value : $TestStruct1 = {#foo = 100000 : i32, #bar = undefined : <1x3x4.f64>, #baz = undefined : [4 x [3 x <3.i32>]]} : $TestStruct1")
    }

    func testWriteSimpleFunction() {
        let fun = builder.buildFunction(named: "foo",
                                        arguments: [ "x" : .scalar(.float(.single)),
                                                     "y" : .scalar(.float(.single)) ],
                                        result: .tensor([3], .bool))
        builder.move(to: fun.entry)
        _ = builder.multiply(.literal(.int(32), 5),
                             .literal(.int(32), .tensor([.literal(.int(32), 1), .literal(.int(32), 2)])),
                             broadcasting: [])
        builder.return(builder.makeLiteral(.null, ofType: .tensor([3], .bool)))
        XCTAssertEqual(fun.description, "func @foo : (f32, f32) -> <3.bool> {\nentry(%x : f32, %y : f32):\n    %v0 = multiply 5 : i32, <1 : i32, 2 : i32> : i32 broadcasting\n    return null : <3.bool>\n}")
    }

    func testWriteMultiBBFunction() {
        let fun = builder.buildFunction(named: "bar",
                                        arguments: [ "x" : .scalar(.float(.single)),
                                                     "y" : .scalar(.float(.single)) ],
                                        result: .int(32))
        builder.move(to: fun.entry)
        let mult = builder.multiply(.literal(.int(32), 5), .literal(.int(32), 8))
        let cmp = builder.compare(.equal, mult, .literal(.int(32), 1))
        let thenBB = builder.buildBasicBlock(named: "then", arguments: [ "x" : .int(32) ], in: fun)
        let elseBB = builder.buildBasicBlock(named: "else", arguments: [ "x" : .int(32) ], in: fun)
        let contBB = builder.buildBasicBlock(named: "cont", arguments: [ "x" : .int(32) ], in: fun)
        builder.conditional(cmp, then: thenBB, arguments: [.literal(.int(32), 0)],
                            else: elseBB, arguments: [.literal(.int(32), 1)])
        builder.move(to: thenBB)
        builder.branch(contBB, [ %thenBB.arguments[0] ])
        builder.move(to: elseBB)
        builder.branch(contBB, [ %elseBB.arguments[0] ])
        builder.move(to: contBB)
        builder.return(%contBB.arguments[0])

        /// func @bar : (f32, f32) -> i32 {
        /// entry(%x : f32, %y : f32):
        ///     %v0 = multiply 5 : i32, 8 : i32
        ///     %v1 = equal %v0 : i32, 1 : i32
        ///     conditional %v1 : bool then then(0 : i32) else else(1 : i32)
        /// then(%x : i32):
        ///     branch cont(%x : i32)
        /// else(%x : i32):
        ///     branch cont(%x : i32)
        /// cont(%x : i32):
        ///     return %x : i32
        /// }
        XCTAssertEqual(fun.description, "func @bar : (f32, f32) -> i32 {\nentry(%x : f32, %y : f32):\n    %v0 = multiply 5 : i32, 8 : i32\n    %v1 = equal %v0 : i32, 1 : i32\n    conditional %v1 : bool then then(0 : i32) else else(1 : i32)\nthen(%x : i32):\n    branch cont(%x : i32)\nelse(%x : i32):\n    branch cont(%x : i32)\ncont(%x : i32):\n    return %x : i32\n}")
    }

    static var allTests : [(String, (IRTests) -> () throws -> Void)] {
        return [
            ("testWriteGlobal", testWriteGlobal),
            ("testWriteStruct", testWriteStruct),
            ("testWriteSimpleFunction", testWriteSimpleFunction),
            ("testWriteMultiBBFunction", testWriteMultiBBFunction)
        ]
    }
    
}
