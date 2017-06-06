import XCTest
@testable import DLVMTests
@testable import DLParseTests

XCTMain([
     testCase(DLVMTests.allTests),
     testCase(DLParseTests.allTests),
])
