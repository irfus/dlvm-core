import XCTest
@testable import CuDNN

class CuDNNTests: XCTestCase {
    
    func testTensorDescriptor() {
        let tensor = TensorDescriptor<Float>(shape: [1, 2, 3, 4, 5, 6, 7, 8])
        XCTAssertEqual(tensor.shape.rank, 8)
        XCTAssertEqual(tensor.shape.dimensions, [1, 2, 3, 4, 5, 6, 7, 8])
        XCTAssertEqual(tensor.shape.strides, [1, 1, 1, 1, 1, 1, 1, 1])
    }
    
    func testTensor() {
        let tensor = Tensor<Float>(shape: [1, 2, 3, 4, 5, 6, 7, 8], repeating: 0.0)
        XCTAssertEqual(tensor.shape.rank, 8)
        XCTAssertEqual(tensor.shape.dimensions, [1, 2, 3, 4, 5, 6, 7, 8])
        XCTAssertEqual(tensor.shape.strides, [1, 1, 1, 1, 1, 1, 1, 1])
        XCTAssertEqual(tensor[0, 1, 2, 3, 4, 5, 6, 7].value, 0)
    }

    static var allTests : [(String, (CuDNNTests) -> () throws -> Void)] {
        return [
        ]
    }
}
