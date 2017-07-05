//
//  CodeGenerator.swift
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

import DLVM
import LLVM_C

/// Code generator protocol
/// - Note: This is often used as a target-erased existential for LLGen
public protocol CodeGenerator {
    func emitIR()
    func writeBitcode(toFile file: String) throws
    var textualIR: String { get }
}

/// LLGen, the unified LLVM code generator for DLVM modules
public class LLGen<TargetType : ComputeTarget> {
    public let dlModule: DLVM.Module
    public lazy internal(set) var context: LLGenContext<TargetType> = LLGenContext(module: self.dlModule)
    var environment: LLGenEnvironment = LLGenEnvironment()

    public init(module: DLVM.Module) {
        self.dlModule = module
        do { try module.verify() }
        catch { DLImpossible() }
    }
}

/// Error thrown during LLGen
public enum LLGenError : Error {
    case fileError(path: String)
}

extension LLGen : CodeGenerator {
    public func emitIR() {
        dlModule.emit(to: context, in: environment)
    }
    
    public func writeBitcode(toFile path: String) throws {
        guard LLVMWriteBitcodeToFile(context.module, path) == 0 else {
            throw LLGenError.fileError(path: path)
        }
    }

    public var textualIR: String {
        let cStr = LLVMPrintModuleToString(context.module) ?? DLImpossibleResult()
        return String(cString: cStr)
    }
}
