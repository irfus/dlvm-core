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
        typealias Expr = Expression<Float>
        let x = Expr.variable(shape: [2, 1], name: "x")
        let W1 = Expr.variable(shape: [2, 2], name: "W1")
        let b1 = Expr.variable(shape: [2, 1], name: "b1")
        let W2 = Expr.variable(shape: [2, 4], name: "W2")
        let b2 = Expr.variable(shape: [4, 1], name: "b2")
        let graph = Graph(expression: softmax(W2 • tanh(W1 • x + b1) + b2))
        XCTAssertEqual(graph.tape, [
            Assignment(variable: "v1", value: .dot("W1", "x")),
            Assignment(variable: "v2", value: .add("v1", "b1")),
            Assignment(variable: "v3", value: .tanh("v2")),
            Assignment(variable: "v4", value: .dot("W2", "v3")),
            Assignment(variable: "v5", value: .add("v4", "b2")),
            Assignment(variable: "v6", value: .softmax("v5"))
        ])
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
