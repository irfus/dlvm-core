//
//  GraphTests.swift
//  DLVM
//
//  Created by Richard Wei on 2/20/17.
//
//

import XCTest
@testable import DLVM

class GraphTests: XCTestCase {
    func testTraversal() {
        let tree = Tree<Int>(value: 1, children: [
            Tree(value: 3, children: [
                Tree(value: 2),
                Tree(value: 4)
            ]),
            Tree(value: 5, children: [
                Tree(value: 10),
                Tree(value: 11),
                Tree(value: 100, children: [
                    Tree(value: 110),
                    Tree(value: 120)
                ])
            ])
        ])
        XCTAssertEqual(tree.preorder.map{$0.value}, [1, 3, 2, 4, 5, 10, 11, 100, 110, 120])
        XCTAssertEqual(tree.levelOrder.map{$0.value}, [1, 3, 5, 2, 4, 10, 11, 100, 110, 120])
    }

    static var allTests : [(String, (GraphTests) -> () throws -> Void)] {
        return [
        ]
    }
}
