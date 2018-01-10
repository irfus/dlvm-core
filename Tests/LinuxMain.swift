import XCTest
@testable import DLVMTests
@testable import DLParseTests

XCTMain([
     // DLVMTests
     testCase(ADTTests.allTests),
     testCase(AnalysisTests.allTests),
     testCase(GraphTests.allTests),
     testCase(IRTests.allTests),
     testCase(TransformTests.allTests),
     // DLParseTests
     testCase(LexTests.allTests),
     testCase(ParseTests.allTests)
])
