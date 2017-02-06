//
//  TensorTests.swift
//  DLVM
//
//  Created by Richard Wei on 2/5/17.
//
//

import XCTest
import struct DLVM.TensorShape
import struct DLVM.TensorIndex
@testable import DLVMRuntime

class TensorTests: XCTestCase {

    func testIndexCalculation() {
        let shape: TensorShape = [1, 2, 3] // 6
        let index: TensorIndex = [0, 0, 1]
        let contIndex = index.contiguousIndex(in: shape)
        XCTAssertEqual(contIndex, 1)
    }

    func testTensor() {
        let shape: TensorShape = [1, 2, 3]
        let tensor = Tensor<Int>(shape: shape, repeating: 0)
        XCTAssertEqual(tensor[].shape, shape)
        XCTAssertEqual(tensor[0].shape, shape.dropFirst())
        XCTAssertEqual(tensor[0, 1].shape, shape.dropFirst(2))
        XCTAssertEqual(tensor[0][1].shape, shape.dropFirst(2))
        XCTAssertEqual(tensor[0, 1, 2].shape, .scalar)
        XCTAssertEqual(tensor[0][1][2].shape, .scalar)
        for subTensor in tensor[0] {
            XCTAssertEqual(subTensor.shape, shape.dropFirst(2))
            XCTAssertEqual(subTensor.items, [0, 0, 0])
        }
    }

    static var allTests : [(String, (TensorTests) -> () throws -> Void)] {
        return [
        ]
    }

}
