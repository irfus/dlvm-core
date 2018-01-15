//
//  IRTests.swift
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
@testable import DLVM

class IRTests : XCTestCase {
    let builder = IRBuilder(moduleName: "IRTest")

    lazy var struct1 = builder.buildStruct(
        named: "TestStruct1", fields: [
            "foo" : .int(32),
            "bar" : .tensor([1, 3, 4], .float(.double)),
            "baz" : .array(4, .array(3, .tensor([3], .int(32))))
        ])

    lazy var struct1Global =
        builder.buildGlobalValue(named: "struct1", type: struct1.type)

    var enum1: EnumType {
        let tmp = builder.buildEnum(
            named: "TestEnum1",
            cases: ["foo" : [.int(32), .float(.single)], "bar" : []])
        tmp.appendCase("baz", with: [.enum(tmp), .tensor([1, 3, 4], .float(.double)), .enum(tmp)])
        return tmp
    }

    lazy var enum1Global: Variable =
        builder.buildGlobalValue(named: "enum1", type: enum1.type)

    func testInitializeStruct() {
        let structLit: Literal = .struct([
            ("foo", 100000 ~ .int(32)),
            ("bar", .undefined ~ .tensor([1, 3, 4], .float(.double))),
            ("baz", .undefined ~ .array(4, .array(3, .tensor([3], .int(32)))))
        ])
        let fun = builder.buildFunction(named: "initialize_struct1",
                                        argumentTypes: [],
                                        returnType: .void)
        builder.move(to: builder.buildEntry(argumentNames: [], in: fun))
        let structInst = builder.literal(structLit, .struct(struct1))
        builder.store(%structInst, to: %struct1Global)
        builder.return()
        XCTAssertEqual(fun.description, """
            func @initialize_struct1: () -> () {
            'entry():
                %0.0 = literal {#foo = 100000: i32, \
            #bar = undefined: <1 x 3 x 4 x f64>, \
            #baz = undefined: [4 x [3 x <3 x i32>]]}: $TestStruct1
                store %0.0: $TestStruct1 to @struct1: *$TestStruct1
                return
            }
            """)
    }

    func testInitializeEnum() {
        let fun = builder.buildFunction(named: "initialize_enum1",
                                        argumentTypes: [],
                                        returnType: .void)
        builder.move(to: builder.buildEntry(argumentNames: [], in: fun))
        let enumInst1 = builder.literal(
            .enumCase("foo", [
                .literal(.int(32), 123), .literal(.float(.single), 3.14)
            ]), .enum(enum1))
        let undefined = builder.literal(.undefined, .tensor([1, 3, 4], .float(.double)))
        let enumInst2 = builder.literal(.enumCase("bar", []), .enum(enum1))
        let enumInst3 = builder.literal(
            .enumCase("baz", [
                %enumInst1, %undefined, %enumInst2
            ]), .enum(enum1))
        builder.store(%enumInst3, to: %enum1Global)
        builder.return()
        XCTAssertEqual(fun.description, """
            func @initialize_enum1: () -> () {
            'entry():
                %0.0 = literal ?foo(123: i32, 3.14: f32): $TestEnum1
                %0.1 = literal undefined: <1 x 3 x 4 x f64>
                %0.2 = literal ?bar(): $TestEnum1
                %0.3 = literal ?baz(%0.0: $TestEnum1, \
            %0.1: <1 x 3 x 4 x f64>, %0.2: $TestEnum1): $TestEnum1
                store %0.3: $TestEnum1 to @enum1: *$TestEnum1
                return
            }
            """)
    }

    func testWriteGlobal() {
        let val1 = builder.buildGlobalValue(named: "one", type: .int(32))
        XCTAssertEqual("\(val1)", "var @one: i32")
        let val2 = builder.buildGlobalValue(named: "two", type: *.int(32))
        XCTAssertEqual("\(val2)", "var @two: *i32")
    }

    func testWriteStruct() {
        XCTAssertEqual(struct1.description, """
            struct $TestStruct1 {
                #foo: i32,
                #bar: <1 x 3 x 4 x f64>,
                #baz: [4 x [3 x <3 x i32>]]
            }
            """)
        XCTAssertEqual("\(struct1Global)", "var @struct1: $TestStruct1")
    }

    func testWriteEnum() {
        XCTAssertEqual(enum1.description, """
            enum $TestEnum1 {
                ?foo(i32, f32),
                ?bar(),
                ?baz($TestEnum1, <1 x 3 x 4 x f64>, $TestEnum1)
            }
            """)
        XCTAssertEqual("\(enum1Global)", "var @enum1: $TestEnum1")
    }

    func testWriteSimpleFunction() {
        let fun = builder.buildFunction(named: "foo",
                                        argumentTypes: [.scalar(.float(.single)), .scalar(.float(.single))],
                                        returnType: .tensor([3], .bool))
        builder.move(to: builder.buildEntry(argumentNames: ["x", "y"], in: fun))
        _ = builder.multiply(.literal(.int(32), 5),
                             .literal(.int(32), .tensor([.literal(.int(32), 1), .literal(.int(32), 2)])))
        builder.return(.null ~ .tensor([3], .bool))
        XCTAssertEqual(fun.description, """
            func @foo: (f32, f32) -> <3 x bool> {
            'entry(%x: f32, %y: f32):
                %0.0 = multiply 5: i32, <1: i32, 2: i32>: i32
                return null: <3 x bool>
            }
            """)
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
        XCTAssertEqual(fun.description, """
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
                return %x: i32
            }
            """)
    }

    static var allTests: [(String, (IRTests) -> () throws -> Void)] {
        return [
            ("testInitializeStruct", testInitializeStruct),
            ("testInitializeEnum", testInitializeEnum),
            ("testWriteGlobal", testWriteGlobal),
            ("testWriteStruct", testWriteStruct),
            ("testWriteEnum", testWriteEnum),
            ("testWriteSimpleFunction", testWriteSimpleFunction),
            ("testWriteMultiBBFunction", testWriteMultiBBFunction)
        ]
    }
}
