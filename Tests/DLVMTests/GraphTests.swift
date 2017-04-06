//
//  GraphTests.swift
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

class GraphTests: XCTestCase {
    func testTraversal() {
        let tree = TreeNode<Int>(value: 1, successors: [
            TreeNode(value: 3, successors: [
                TreeNode(value: 2),
                TreeNode(value: 4)
            ]),
            TreeNode(value: 5, successors: [
                TreeNode(value: 10),
                TreeNode(value: 11),
                TreeNode(value: 100, successors: [
                    TreeNode(value: 110),
                    TreeNode(value: 120)
                ])
            ])
        ])
        XCTAssertEqual(tree.preorder.map{$0.value}, [1, 3, 2, 4, 5, 10, 11, 100, 110, 120])
        XCTAssertEqual(tree.postorder.map{$0.value}, [2, 4, 3, 10, 11, 110, 120, 100, 5, 1])
        XCTAssertEqual(tree.breadthFirst.map{$0.value}, [1, 3, 5, 2, 4, 10, 11, 100, 110, 120])
    }

    static var allTests : [(String, (GraphTests) -> () throws -> Void)] {
        return [
        ]
    }
}
