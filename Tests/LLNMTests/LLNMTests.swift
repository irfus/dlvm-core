import XCTest
@testable import LLNM

class LLNMTests: XCTestCase {
    
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

    func testBuildTape() {
        let x = Expression<Float>.tensor(shape: [1, 2], name: "x")
        let W1 = Expression<Float>.tensor(shape: [1, 2], name: "W1")
        let b = Expression<Float>.tensor(shape: [1, 2], name: "b1")
        let l1 = tanh(W1 • x + b)
        let graph = ExpressionGraph(expression: l1)
        for assn in graph.tape {
            print(assn)
        }
    }

    static var allTests : [(String, (LLNMTests) -> () throws -> Void)] {
        return [
            ("testTensorDescriptor", testTensorDescriptor),
            ("testTensor", testTensor),
            ("testBuildTape", testBuildTape)
        ]
    }
}
