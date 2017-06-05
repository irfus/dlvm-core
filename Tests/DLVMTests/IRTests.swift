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
                                            initializer: %InstructionKind.zipWith(
                                                .associative(.add),
                                                .literal(.int(32), .scalar(.int(10))),
                                                .literal(.int(32), .scalar(.int(20)))))
        XCTAssertEqual("\(val1)", "let @one: i32 = (add 10: i32, 20: i32): i32")
        let val2 = builder.buildGlobalValue(named: "two", kind: .constant,
                                            type: Type.int(32).pointer,
                                            initializer: %val1)
        XCTAssertEqual("\(val2)", "let @two: *i32 = @one: *i32")
    }

    func testWriteStruct() {
        let struct1 = builder.buildStruct(named: "TestStruct1", fields: [
            "foo" : .int(32),
            "bar" : .tensor([1, 3, 4], .float(.double)),
            "baz" : .array(4, .array(3, .tensor([3], .int(32))))
        ])
        XCTAssertEqual(struct1.description,
                       "struct $TestStruct1 {\n    #foo: i32\n    #bar: <1 x 3 x 4 x f64>\n    #baz: [4 x [3 x <3 x i32>]]\n}")
        let structLit : Literal = .struct([
            ("foo", 100000 ~ .int(32)),
            ("bar", .undefined ~ .tensor([1, 3, 4], .float(.double))),
            ("baz", .undefined ~ .array(4, .array(3, .tensor([3], .int(32)))))
        ])
        let structGlobal = builder.buildGlobalValue(named: "struct1.value",
                                                    kind: .variable,
                                                    type: struct1.type,
                                                    initializer: structLit ~ struct1.type)
        XCTAssertEqual("\(structGlobal)", "var @struct1.value: $TestStruct1 = {#foo = 100000: i32, #bar = undefined: <1 x 3 x 4 x f64>, #baz = undefined: [4 x [3 x <3 x i32>]]}: $TestStruct1")
    }

    func testWriteSimpleFunction() {
        let fun = builder.buildFunction(named: "foo",
                                        argumentTypes: [.scalar(.float(.single)), .scalar(.float(.single))],
                                        returnType: .tensor([3], .bool))
        builder.move(to: builder.buildEntry(argumentNames: ["x", "y"], in: fun))
        _ = builder.multiply(.literal(.int(32), 5),
                             .literal(.int(32), .tensor([.literal(.int(32), 1), .literal(.int(32), 2)])))
        builder.return(.null ~ .tensor([3], .bool))
        XCTAssertEqual(fun.description, "func @foo: (f32, f32) -> <3 x bool> {\n'entry(%x: f32, %y: f32):\n    %0.0 = multiply 5: i32, <1: i32, 2: i32>: i32\n    return null: <3 x bool>\n}")
    }

    func testWriteMultiBBFunction() {
        let fun = builder.buildFunction(named: "bar",
                                        argumentTypes: [.scalar(.float(.single)), .scalar(.float(.single))],
                                        returnType: .int(32))
        builder.move(to: builder.buildEntry(argumentNames: ["x", "y"], in: fun))
        let mult = builder.multiply(.literal(.int(32), 5), .literal(.int(32), 8))
        let cmp = builder.compare(.equal, %mult, .literal(.int(32), 1))
        let thenBB = builder.buildBasicBlock(named: "then", arguments: [ "x" : .int(32) ], in: fun)
        let elseBB = builder.buildBasicBlock(named: "else", arguments: [ "x" : .int(32) ], in: fun)
        let contBB = builder.buildBasicBlock(named: "cont", arguments: [ "x" : .int(32) ], in: fun)
        builder.conditional(%cmp, then: thenBB, arguments: [.literal(.int(32), 0)],
                            else: elseBB, arguments: [.literal(.int(32), 1)])
        builder.move(to: thenBB)
        builder.branch(contBB, [ %thenBB.arguments[0] ])
        builder.move(to: elseBB)
        builder.branch(contBB, [ %elseBB.arguments[0] ])
        builder.move(to: contBB)
        builder.return(%contBB.arguments[0])

        /// func @bar : (f32, f32) -> i32 {
        /// 'entry(%x : f32, %y : f32):
        ///     %0.0 = multiply 5 : i32, 8 : i32
        ///     %0.1 = equal %v0 : i32, 1 : i32
        ///     conditional %v1 : bool then 'then(0 : i32) else 'else(1 : i32)
        /// 'then(%x : i32):
        ///     branch 'cont(%x : i32)
        /// 'else(%x : i32):
        ///     branch 'cont(%x : i32)
        /// 'cont(%x : i32):
        ///     return %x : i32
        /// }
        XCTAssertEqual(fun.description, "func @bar: (f32, f32) -> i32 {\n'entry(%x: f32, %y: f32):\n    %0.0 = multiply 5: i32, 8: i32\n    %0.1 = equal %0.0: i32, 1: i32\n    conditional %0.1: bool then 'then(0: i32) else 'else(1: i32)\n'then(%x: i32):\n    branch 'cont(%x: i32)\n'else(%x: i32):\n    branch 'cont(%x: i32)\n'cont(%x: i32):\n    return %x: i32\n}")
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
