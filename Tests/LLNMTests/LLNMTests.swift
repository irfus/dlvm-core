import XCTest
@testable import LLNM

class LLNMTests: XCTestCase {

    lazy var graph1: Graph<Float> = {
        let x = Expression<Float>.input(shape: [2, 1], name: "x")
        let W1 = Expression<Float>.parameter(shape: [2, 2], initial: .zeros, name: "W1")
        let b1 = Expression<Float>.parameter(shape: [2, 1], initial: .zeros, name: "b1")
        let W2 = Expression<Float>.parameter(shape: [4, 2], initial: .zeros, name: "W2")
        let b2 = Expression<Float>.parameter(shape: [4, 1], initial: .zeros, name: "b2")

        let l1 = tanh(W1 • x + b1) ~ "l1"
        let l2 = softmax(W2 • (1 - l1) + b2) ~ "l2"

        return try! Graph<Float>(expression: l2)
    }()
    
    func testTensorDescriptor() {
        let tensor1 = TensorDescriptor<Float>(shape: [1, 2])
        XCTAssertEqual(tensor1.shape.rank, 2)
        XCTAssertEqual(tensor1.shape.dimensions, [1, 2])
        let tensor2 = TensorDescriptor<Float>(shape: [1, 2, 3, 4, 5, 6, 7, 8])
        XCTAssertEqual(tensor2.shape.rank, 8)
        XCTAssertEqual(tensor2.shape.dimensions, [1, 2, 3, 4, 5, 6, 7, 8])
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

        let l1 = tanh(W1 • x + b1) ~ "l1"
        let l2 = softmax(W2 • (1 - l1) + b2) ~ "l2"

        let graph = try Graph<Float>(expression: l2)

        print(graph)
    }

    func testTensorOp() throws {
        let x = Expression<Float>.parameter(shape: [2, 1],
                                            initial: .random(from: 0.0, to: 1.0),
                                            name: "x")
        let y = Expression<Float>.parameter(shape: [2, 2],
                                            initial: .random(from: 0.0, to: 1.0),
                                            name: "y")
        let o = Expression.product(y, x)
        let graph = try Graph<Float>(expression: o)
        let assignment = graph.tape.last!
        assignment.propagateForward()

        /// Whole graph
        for variable in graph1.tape {
            print(variable)
            variable.propagateForward()
            print(variable.data.elements.hostArray)
        }
    }

    static var allTests : [(String, (LLNMTests) -> () throws -> Void)] {
        return [
            ("testTensorDescriptor", testTensorDescriptor),
            ("testTensor", testTensor),
            ("testBuildTape", testBuildTape),
            ("testTensorOp", testTensorOp)
        ]
    }
}
