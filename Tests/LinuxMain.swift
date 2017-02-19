import XCTest
@testable import DLVMTests
@testable import DLVMRuntimeTests
@testable import TELTests

XCTMain([
     testCase(DLVMTests.allTests),
     testCase(DLVMRuntimeTests.allTests),
     testCase(TELTests.allTests),
])
