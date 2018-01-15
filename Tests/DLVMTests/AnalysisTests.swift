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
        let fun = builder.buildFunction(
            named: "gcd",
            argumentTypes: [.scalar(.int(32)), .scalar(.int(32))],
            returnType: .int(32))
        let entry = builder.buildEntry(argumentNames: ["a", "b"], in: fun)
        let a = entry.arguments[0]
        let b = entry.arguments[1]
        builder.move(to: entry)
        let cmp = builder.compare(
            .equal, %entry.arguments[1], .literal(.int(32), 0))
        let thenBB = builder.buildBasicBlock(
            named: "then", arguments: [:], in: fun)
        let elseBB = builder.buildBasicBlock(
            named: "else", arguments: [:], in: fun)
        builder.conditional(%cmp, then: thenBB, arguments: [],
                            else: elseBB, arguments: [])
        builder.move(to: thenBB)
        builder.return(%a)
        builder.move(to: elseBB)
        let t = builder.modulo(%a, %b)
        builder.branch(entry, [%b, %t])
        XCTAssertEqual(fun.description, """
            func @gcd: (i32, i32) -> i32 {
            'entry(%a: i32, %b: i32):
                %0.0 = equal %b: i32, 0: i32
                conditional %0.0: bool then 'then() else 'else()
            'then():
                return %a: i32
            'else():
                %2.0 = modulo %a: i32, %b: i32
                branch 'entry(%b: i32, %2.0: i32)
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
            XCTAssertEqual(loop.blocks, [entry, elseBB])
            XCTAssertEqual(loopInfo.innerMostLoops,
                           [entry : loop, elseBB : loop])
        }
    }

    func testNestedLoop() {
        let fun = builder.buildFunction(
            named: "double_loop",
            argumentTypes: [.scalar(.int(32)), .scalar(.int(32))],
            returnType: .int(32))
        let entry = builder.buildEntry(argumentNames: ["a", "b"], in: fun)
        builder.move(to: entry)
        let outerBB = builder.buildBasicBlock(
            named: "outer",
            arguments: ["i1" : .int(32), "j1" : .int(32)], in: fun)
        let innerBB = builder.buildBasicBlock(
            named: "inner",
            arguments: ["i2" : .int(32), "j2" : .int(32)], in: fun)
        let iBodyBB = builder.buildBasicBlock(
            named: "inner_body",
            arguments: [:], in: fun)
        let oBodyBB = builder.buildBasicBlock(
            named: "outer_body",
            arguments: [:], in: fun)
        let contBB = builder.buildBasicBlock(
            named: "cont", arguments: [:], in: fun)
        builder.branch(outerBB, [.literal(.int(32), 0), .literal(.int(32), 0)])
        builder.move(to: outerBB)
        let outerArgs = outerBB.arguments.map { arg in %arg }
        let outerCond = builder.compare(.lessThan, outerArgs[0],
                                        %entry.arguments[0])
        builder.conditional(%outerCond, then: innerBB, arguments: outerArgs,
                            else: contBB, arguments: [])
        builder.move(to: innerBB)
        let innerArgs = innerBB.arguments.map { arg in %arg }
        let innerCond = builder.compare(.lessThan, innerArgs[1],
                                        %entry.arguments[1])
        builder.conditional(%innerCond, then: iBodyBB, arguments: [],
                            else: oBodyBB, arguments: [])
        builder.move(to: iBodyBB)
        let jIncr = builder.add(innerArgs[1], .literal(.int(32), 1))
        builder.branch(innerBB, [innerArgs[0], %jIncr])
        builder.move(to: oBodyBB)
        let iIncr = builder.add(innerArgs[0], .literal(.int(32), 1))
        builder.branch(outerBB, [%iIncr, innerArgs[1]])
        builder.move(to: contBB)
        let sum = builder.add(outerArgs[0], outerArgs[1])
        builder.return(%sum)
        XCTAssertEqual(fun.description, """
            func @double_loop: (i32, i32) -> i32 {
            'entry(%a: i32, %b: i32):
                branch 'outer(0: i32, 0: i32)
            'outer(%i1: i32, %j1: i32):
                %1.0 = lessThan %i1: i32, %a: i32
                conditional %1.0: bool then 'inner(%i1: i32, %j1: i32) else 'cont()
            'inner(%i2: i32, %j2: i32):
                %2.0 = lessThan %j2: i32, %b: i32
                conditional %2.0: bool then 'inner_body() else 'outer_body()
            'inner_body():
                %3.0 = add %j2: i32, 1: i32
                branch 'inner(%i2: i32, %3.0: i32)
            'outer_body():
                %4.0 = add %i2: i32, 1: i32
                branch 'outer(%4.0: i32, %j2: i32)
            'cont():
                %5.0 = add %i1: i32, %j1: i32
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
