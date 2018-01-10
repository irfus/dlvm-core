//
//  AnalysisTests.swift
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

class AnalysisTests : XCTestCase {
    let builder = IRBuilder(moduleName: "AnalysisTest")

    func testLoop() {
        let fun = builder.buildFunction(named: "gcd",
                                        argumentTypes: [.scalar(.int(32)), .scalar(.int(32))],
                                        returnType: .int(32))
        let entry = builder.buildEntry(argumentNames: ["a", "b"], in: fun)
        builder.move(to: entry)
        let cmp = builder.compare(.equal, %entry.arguments[1], .literal(.int(32), 0))
        let nextBB = builder.buildBasicBlock(named: "next", arguments: [:], in: fun)
        let thenBB = builder.buildBasicBlock(named: "then", arguments: ["a" : .int(32)], in: fun)
        let elseBB = builder.buildBasicBlock(named: "else", arguments: ["a" : .int(32), "b" : .int(32)], in: fun)
        let contBB = builder.buildBasicBlock(named: "cont", arguments: ["a" : .int(32)], in: fun)
        builder.branch(nextBB, [])
        builder.move(to: nextBB)
        builder.conditional(%cmp, then: thenBB, arguments: [%entry.arguments[0]],
                            else: elseBB, arguments: [%entry.arguments[0], %entry.arguments[1]])
        builder.move(to: thenBB)
        builder.branch(contBB, [%thenBB.arguments[0]])
        builder.move(to: elseBB)
        let t = builder.modulo(%elseBB.arguments[0], %elseBB.arguments[1])
        builder.branch(entry, [%elseBB.arguments[1], %t])
        builder.move(to: contBB)
        builder.return(%contBB.arguments[0])
        XCTAssertEqual(fun.description, """
            func @gcd: (i32, i32) -> i32 {
            'entry(%a: i32, %b: i32):
                %0.0 = equal %b: i32, 0: i32
                branch 'next()
            'next():
                conditional %0.0: bool then 'then(%a: i32) else 'else(%a: i32, %b: i32)
            'then(%a: i32):
                branch 'cont(%a: i32)
            'else(%a: i32, %b: i32):
                %3.0 = modulo %a: i32, %b: i32
                branch 'entry(%b: i32, %3.0: i32)
            'cont(%a: i32):
                return %a: i32
            }
            """)
        /// Test multiple times to account for nondeterminstic behavior
        for _ in 0...100 {
            /// Check back edges
            let backEdges = fun.backEdges(fromEntry: entry)
            XCTAssert(backEdges == [(elseBB, entry)])
            /// Check dominance
            let domTree = fun.analysis(from: DominanceAnalysis.self)
            for bb in fun {
                for succ in domTree.successors(of: bb) {
                    XCTAssertEqual(domTree.predecessors(of: succ), [bb])
                }
            }
            /// Check loop info
            let loopInfo = fun.analysis(from: LoopAnalysis.self)
            XCTAssertEqual(loopInfo.topLevelLoops.count, 1)
            let loop = loopInfo.topLevelLoops[0]
            XCTAssertEqual(loop.header, entry)
            XCTAssertEqual(loop.subloops, [])
            XCTAssertEqual(loop.blocks, [entry, nextBB, elseBB])
            XCTAssertEqual(loopInfo.innerMostLoops, [entry : loop, nextBB : loop, elseBB : loop])
        }
    }

    func testNestedLoop() {
        let fun = builder.buildFunction(named: "double_loop",
                                        argumentTypes: [.scalar(.int(32)), .scalar(.int(32))],
                                        returnType: .int(32))
        let entry = builder.buildEntry(argumentNames: ["a", "b"], in: fun)
        builder.move(to: entry)
        let i = builder.literal(0, .int(32))
        let j = builder.literal(0, .int(32))
        let outerBB = builder.buildBasicBlock(named: "outer", arguments: ["i" : .int(32), "j" : .int(32)], in: fun)
        let innerBB = builder.buildBasicBlock(named: "inner", arguments: ["i" : .int(32), "j" : .int(32)], in: fun)
        let iBodyBB = builder.buildBasicBlock(named: "inner_body", arguments: ["i" : .int(32), "j" : .int(32)], in: fun)
        let oBodyBB = builder.buildBasicBlock(named: "outer_body", arguments: ["i" : .int(32), "j" : .int(32)], in: fun)
        let contBB = builder.buildBasicBlock(named: "cont", arguments: ["i" : .int(32), "j" : .int(32)], in: fun)
        builder.branch(outerBB, [%i, %j])
        builder.move(to: outerBB)
        let outerArgs = outerBB.arguments.map { arg in %arg }
        let outerCond = builder.compare(.lessThan, outerArgs[0], %entry.arguments[0])
        builder.conditional(%outerCond, then: innerBB, arguments: outerArgs,
                            else: contBB, arguments: outerArgs)
        builder.move(to: innerBB)
        let innerArgs = innerBB.arguments.map { arg in %arg }
        let innerCond = builder.compare(.lessThan, innerArgs[1], %entry.arguments[1])
        builder.conditional(%innerCond, then: iBodyBB, arguments: innerArgs,
                            else: oBodyBB, arguments: innerArgs)
        builder.move(to: iBodyBB)
        let jIncr = builder.add(%iBodyBB.arguments[1], .literal(.int(32), 1))
        builder.branch(innerBB, [%iBodyBB.arguments[0], %jIncr])
        builder.move(to: oBodyBB)
        let iIncr = builder.add(%iBodyBB.arguments[0], .literal(.int(32), 1))
        builder.branch(outerBB, [%iIncr, %oBodyBB.arguments[1]])
        builder.move(to: contBB)
        let sum = builder.add(%contBB.arguments[0], %contBB.arguments[1])
        builder.return(%sum)
        XCTAssertEqual(fun.description, """
            func @double_loop: (i32, i32) -> i32 {
            'entry(%a: i32, %b: i32):
                %0.0 = literal 0: i32
                %0.1 = literal 0: i32
                branch 'outer(%0.0: i32, %0.1: i32)
            'outer(%i: i32, %j: i32):
                %1.0 = lessThan %i: i32, %a: i32
                conditional %1.0: bool then 'inner(%i: i32, %j: i32) else 'cont(%i: i32, %j: i32)
            'inner(%i: i32, %j: i32):
                %2.0 = lessThan %j: i32, %b: i32
                conditional %2.0: bool then 'inner_body(%i: i32, %j: i32) else 'outer_body(%i: i32, %j: i32)
            'inner_body(%i: i32, %j: i32):
                %3.0 = add %j: i32, 1: i32
                branch 'inner(%i: i32, %3.0: i32)
            'outer_body(%i: i32, %j: i32):
                %4.0 = add %i: i32, 1: i32
                branch 'outer(%4.0: i32, %j: i32)
            'cont(%i: i32, %j: i32):
                %5.0 = add %i: i32, %j: i32
                return %5.0: i32
            }
            """)
        /// Test multiple times to account for nondeterminstic behavior
        for _ in 0...100 {
            /// Check back edges
            let backEdges = fun.backEdges(fromEntry: entry)
                .sorted(by: {$0.0.indexInParent < $1.0.indexInParent})
            XCTAssert(backEdges == [(iBodyBB, innerBB), (oBodyBB, outerBB)])
            /// Check dominance
            let domTree = fun.analysis(from: DominanceAnalysis.self)
            for bb in fun {
                for succ in domTree.successors(of: bb) {
                    XCTAssertEqual(domTree.predecessors(of: succ), [bb])
                }
            }
            /// Check loop info
            let loopInfo = fun.analysis(from: LoopAnalysis.self)
            XCTAssertEqual(loopInfo.topLevelLoops.count, 1)
            let loop = loopInfo.topLevelLoops[0]
            XCTAssertEqual(loop.header, outerBB)
            XCTAssertEqual(loop.subloops.count, 1)
            let subloop = loop.subloops.first
            XCTAssertEqual(subloop?.header, innerBB)
            XCTAssertEqual(subloop?.subloops, [])
            XCTAssertEqual(subloop?.blocks, [innerBB, iBodyBB])
            XCTAssert(loop.blocks == [outerBB, innerBB, iBodyBB, oBodyBB] ||
                loop.blocks == [outerBB, innerBB, oBodyBB, iBodyBB])
            XCTAssertEqual(loopInfo.innerMostLoops,
                           [outerBB : loop, oBodyBB : loop,
                            innerBB : subloop, iBodyBB : subloop])
        }
    }

    static var allTests: [(String, (AnalysisTests) -> () throws -> Void)] {
        return [
            ("testLoop", testLoop),
            ("testNestedLoop", testNestedLoop)
        ]
    }
}
