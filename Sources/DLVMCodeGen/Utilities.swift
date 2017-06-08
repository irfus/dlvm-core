//
//  Utilities.swift
//  DLVM
//
//  Copyright 2016-2017 Richard Wei.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

import LLVM_C

// MARK: - Impossible case crasher
/// - Note: This is called when LLGen encounters ill-formed DLVM IR.
/// All well-formedness should be checked by the module verifier, not by LLGen
@discardableResult
func DLImpossibleResult<T>(function: String = #function,
                           file: StaticString = #file,
                           line: UInt = #line) -> T {
    fatalError("Impossible case \(T.self). Something's wrong before LLGen")
}

func DLImpossible(function: String = #function,
                  file: StaticString = #file,
                  line: UInt = #line) -> Never {
    let _: () = DLImpossibleResult(function: function, file: file, line: line)
    fatalError()
}

func DLAssert(_ condition: @autoclosure () -> Bool,
              function: String = #function,
              file: StaticString = #file,
              line: UInt = #line) {
    guard condition() else {
        DLImpossible(function: function, file: file, line: line)
    }
}

import Foundation

func environmentVariable(named name: String) -> String? {
    return ProcessInfo.processInfo.environment[name]
}

func DLUnimplemented(_ function: String = #function, file: StaticString = #file, line: UInt = #line) -> Never {
    fatalError("\(function) is not fully implemented. \(file):\(line)")
}
