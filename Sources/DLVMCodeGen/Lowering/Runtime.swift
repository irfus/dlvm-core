//
//  Runtime.swift
//  DLVM
//
//  Copyright 2016-2017 The DLVM Team.
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
import LLVM_C

var runtimePath: String {
    guard let path = environmentVariable(named: "DLRT_PATH") else {
        preconditionFailure("DLVM runtime bitcode path (DLRT_PATH) not defined")
    }
    return path
}

var runtimeModule: LLVMModuleRef {
    func checkStatus(_ status: LLVMBool, errorMessageBuffer errorBuf: UnsafeMutablePointer<CChar>?) {
        guard status == 0 else {
            if let error = errorBuf.flatMap ({ String(cString: $0) }) {
                preconditionFailure(error)
            } else {
                preconditionFailure("Cannot read DLVM runtime bitcode module")
            }
        }
    }

    var memoryBuffer: LLVMMemoryBufferRef?
    var errorBuf: UnsafeMutablePointer<CChar>?
    let readStatus = LLVMCreateMemoryBufferWithContentsOfFile(runtimePath, &memoryBuffer, &errorBuf)
    checkStatus(readStatus, errorMessageBuffer: errorBuf)
    
    var module: LLVMModuleRef?
    let parseStatus = LLVMParseBitcode(memoryBuffer, &module, &errorBuf)
    checkStatus(parseStatus, errorMessageBuffer: errorBuf)

    return module!
}
