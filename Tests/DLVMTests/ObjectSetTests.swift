//
//  ObjectSetTests.swift
//  DLVM
//
//  Created by Richard Wei on 2/4/17.
//
//

import XCTest
@testable import struct DLVM.NamedObjectSet
@testable import struct DLVM.OrderedNamedObjectSet

class ObjectSetTests: XCTestCase {

    func testCopyOnWrite() {
        var set = NamedObjectSet<Int>()
        set.insert(1)

        var setCopy = set
        setCopy.insert(2)

        XCTAssertTrue(set.contains(1))
        XCTAssertFalse(set.contains(2))
        XCTAssertTrue(setCopy.contains(1))
        XCTAssertTrue(setCopy.contains(2))
    }

    func testPerformanceExample() {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }

}
