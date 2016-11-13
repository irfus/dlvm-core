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
        XCTAssertEqual(graph.tape, [
            Assignment(variable: "v1", value: .matMul("W1", "x")),
            Assignment(variable: "v2", value: .add("v1", "b1")),
            Assignment(variable: "v3", value: .tanh("v2"))
        ])
    }

    static var allTests : [(String, (LLNMTests) -> () throws -> Void)] {
        return [
            ("testTensorDescriptor", testTensorDescriptor),
            ("testTensor", testTensor),
            ("testBuildTape", testBuildTape)
        ]
    }
}
