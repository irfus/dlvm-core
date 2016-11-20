import XCTest
@testable import LLNM

class LLNMTests: XCTestCase {
    
    func testTensorDescriptor() {
        let tensor = TensorDescriptor<Float>(shape: [1, 2, 3, 4, 5, 6, 7, 8])
        XCTAssertEqual(tensor.shape.rank, 8)
        XCTAssertEqual(tensor.shape.dimensions, [1, 2, 3, 4, 5, 6, 7, 8])
    }
    
    func testTensor() {
        let tensor = Tensor<Float>(shape: [1, 2, 3, 4, 5, 6, 7, 8], repeating: 0.0)
        XCTAssertEqual(tensor.shape.rank, 8)
        XCTAssertEqual(tensor.shape.dimensions, [1, 2, 3, 4, 5, 6, 7, 8])
        XCTAssertEqual(tensor[0, 1, 2, 3, 4, 5, 6, 7].value, 0)
    }

    func testBuildTape() throws {
        let x = Expression<Float>.input(shape: [2, 1], name: "x")
        let W1 = Expression<Float>.parameter(shape: [2, 2], initial: .zeros, name: "W1")
        let b1 = Expression<Float>.parameter(shape: [2, 1], initial: .zeros, name: "b1")
        let W2 = Expression<Float>.parameter(shape: [4, 2], initial: .zeros, name: "W2")
        let b2 = Expression<Float>.parameter(shape: [4, 1], initial: .zeros, name: "b2")

        let l1 = tanh(W1 • x + b1) <- "l1"
        let l2 = softmax(W2 • l1 + b2) <- "output"

        let graph = try Graph<Float>(expression: l2)

        print(graph)
    }

    static var allTests : [(String, (LLNMTests) -> () throws -> Void)] {
        return [
            ("testTensorDescriptor", testTensorDescriptor),
            ("testTensor", testTensor),
            ("testBuildTape", testBuildTape)
        ]
    }
}
