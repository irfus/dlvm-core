//
//  Utilities.swift
//  DLVM
//
//  Copyright 2016-2018 The DLVM Team.
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

import Foundation

func DLUnimplemented(_ function: String = #function,
                     file: StaticString = #file,
                     line: UInt = #line) -> Never {
    fatalError("\(function) is not fully implemented.", file: file, line: line)
}

func DLImpossible(_ function: String = #function,
                  file: StaticString = #file,
                  line: UInt = #line) -> Never {
    fatalError("Impossible case at \(function). Must be a compiler bug.",
               file: file, line: line)
}
