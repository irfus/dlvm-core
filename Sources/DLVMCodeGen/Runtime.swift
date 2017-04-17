//
//  Runtime.swift
//  DLVM
//
//  Created by Richard Wei on 4/16/17.
//
//

import Foundation
import LLVM

public var runtimePath: String {
    guard let path = environmentVariable(named: "DLRT_PATH") else {
        preconditionFailure("DLVM runtime bitcode path (DLRT_PATH) not defined")
    }
    return path
}

import cllvm

public var runtimeModule: LLVMModuleRef {
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
