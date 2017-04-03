import XCTest
@testable import DLVMTests
@testable import TELTests

XCTMain([
     testCase(DLVMTests.allTests),
     testCase(TELTests.allTests),
])
