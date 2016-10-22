import XCTest
@testable import CuDNN

class CuDNNTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct results.
        XCTAssertEqual(CuDNN().text, "Hello, World!")
    }


    static var allTests : [(String, (CuDNNTests) -> () throws -> Void)] {
        return [
            ("testExample", testExample),
        ]
    }
}
