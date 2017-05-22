//
//  TransformTests.swift
//  DLVM
//
//  Created by Richard Wei on 4/27/17.
//
//

import XCTest
@testable import DLVM

class TransformTests: XCTestCase {

    let builder = IRBuilder(moduleName: "TransformTest")

    /// - TODO: Fix bug in dominance analysis that causes crash
    func testDCE() throws {
        let fun = builder.buildFunction(named: "bar",
                                        arguments: [ "x" : .scalar(.float(.single)),
                                                     "y" : .scalar(.float(.single)) ],
                                        result: .int(32))
        builder.move(to: builder.buildEntry(in: fun))
        let mult = builder.multiply(.literal(.int(32), 5), .literal(.int(32), 8))
        let dead1 = builder.buildInstruction(.binary(.associative(.arithmetic(.multiply)),
                                         .literal(.int(32), 10000), .literal(.int(32), 20000), nil),
                                 name: "dead1")
        builder.buildInstruction(.binary(.associative(.arithmetic(.add)),
                                         %dead1, 20000 ~ Type.int(32), nil),
                                 name: "dead2")
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

        /// Original:
        /// func @bar: (f32, f32) -> i32 {
        /// entry(%x: f32, %y : f32):
        ///     %0.0 = multiply 5: i32, 8: i32
        ///     %dead1 = multiply 10000: i32, 20000: i32
        ///     %dead2 = add %dead1: i32, 20000: i32
        ///     %0.3 = equal %v0: i32, 1: i32
        ///     conditional %v1: bool then then(0: i32) else else(1: i32)
        /// then(%x: i32):
        ///     branch cont(%x: i32)
        /// else(%x: i32):
        ///     branch cont(%x: i32)
        /// cont(%x: i32):
        ///     return %x: i32
        /// }

        /// After: 
        /// func @bar: (f32, f32) -> i32 {
        /// entry(%x: f32, %y: f32):
        ///     %0.0 = multiply 5: i32, 8: i32
        ///     %0.1 = equal %v0: i32, 1: i32
        ///     conditional %v1: bool then then(0: i32) else else(1: i32)
        /// then(%x: i32):
        ///     branch cont(%x: i32)
        /// else(%x: i32):
        ///     branch cont(%x: i32)
        /// cont(%x: i32):
        ///     return %x: i32
        /// }
        try builder.module.mapTransform(DeadCodeElimination.self)
        let after = "func @bar: (f32, f32) -> i32 {\nentry(%x: f32, %y: f32):\n    %0.0 = multiply 5: i32, 8: i32\n    %0.1 = equal %0.0: i32, 1: i32\n    conditional %0.1: bool then then(0: i32) else else(1: i32)\nthen(%x: i32):\n    branch cont(%x: i32)\nelse(%x: i32):\n    branch cont(%x: i32)\ncont(%x: i32):\n    return %x: i32\n}"
        XCTAssertEqual(fun.description, after)
        /// Reapplying shouldn't mutate the function

        let module = builder.module
        try module.mapTransform(DeadCodeElimination.self)
        try module.mapTransform(DeadCodeElimination.self)

        XCTAssertEqual(fun.description, after)
    }

    static var allTests : [(String, (TransformTests) -> () throws -> Void)] {
        return [
            ("testDCE", testDCE),
        ]
    }

}
