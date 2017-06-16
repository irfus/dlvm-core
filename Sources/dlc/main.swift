//
//  main.swift
//  dlc
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
import DLParse
import Foundation
import CommandLineKit
import DLVMCodeGen
import DLCommandLineTools

enum Target : String {
    case hpvm = "hpvm"
    case nvptx = "nvptx"
    case amdgpu = "amdgpu"
    case cpu = "cpu"
}

let cli = CommandLineKit.CommandLine()

struct Options {
    /// File
    static let filePaths = MultiStringOption(shortFlag: "f", longFlag: "files",
                                            helpMessage: "Paths to DLVM IR source files")
    /// Print IR
    static let shouldPrintIR = BoolOption(longFlag: "print-ir",
                                          helpMessage: "Print IR after transformation")

    /// Transform passes
    static let passes = MultiStringOption(shortFlag: "p", longFlag: "passes",
                                          helpMessage: "Run passes")

    /// Output
    static let outputPaths = MultiStringOption(shortFlag: "o", longFlag: "outputs",
                                               helpMessage: "Output file paths")
    /// Help
    static let needsHelp = BoolOption(shortFlag: "h", longFlag: "help",
                                      helpMessage: "Print help message")

    /// Target compute back-end
    static let target = EnumOption<Target>(shortFlag: "t", longFlag: "target",
        helpMessage: "Target compute backend [ hpvm | nvptx | amdgpu | cpu ]")

    /// Compile to LLVM
    static let shouldCompile = BoolOption(shortFlag: "c",
                                          helpMessage: "Compile to LLVM")

    /// Emit LLVM IR
    static let shouldEmitLLVM = BoolOption(longFlag: "emit-llvm",
                                           helpMessage: "Emit LLVM Textual IR")

    /// Emit LLVM Bitcode
    static let shouldEmitBitcode = BoolOption(longFlag: "emit-bc",
                                              helpMessage: "Emit LLVM Bitcode")

}

cli.addOptions(Options.filePaths,
               Options.shouldPrintIR,
               Options.passes,
               Options.outputPaths,
               Options.needsHelp,
               Options.target,
               Options.shouldCompile,
               Options.shouldEmitLLVM,
               Options.shouldEmitBitcode)

/// Parse command line
do { try cli.parse(strict: true) }
catch { cli.printUsage(error); exit(EXIT_FAILURE) }

func codeGenerator(for target: Target, from module: Module) -> CodeGenerator {
    switch target {
    case .nvptx:
        return LLGen<NVVM>(module: module)
    case .hpvm:
        return LLGen<HPVM>(module: module)
    case .amdgpu:
        error("AMDGPU target is not yet supported")
    case .cpu:
        error("Pure CPU target is not yet supported")
    }
}

/// Command line entry
func main() throws {
    
    guard !Options.needsHelp.wasSet else {
        print("DLVM IR Compiler\n")
        cli.printUsage()
        return
    }

    guard let filePaths = Options.filePaths.value else {
        error("no input files; use -f to specify files")
    }

    let outputPaths = Options.outputPaths.value
    if let outputPaths = outputPaths {
        guard outputPaths.count == filePaths.count else {
            error("different numbers of inputs and outputs specified")
        }
    }

    for (i, filePath) in filePaths.enumerated() {
        /// Read IR and verify
        print("Source file:", filePath)
        /// Parse
        let module = try Module.parsed(fromFile: filePath)

        /// Run passes
        if let passes = Options.passes.value {
            for passName in passes {
                try runPass(named: passName, on: module)
            }
        }
        
        /// Print IR if requested
        if Options.shouldPrintIR.wasSet {
            print(module)
        }
        
        /// Write transformed IR if requested
        if let outputPaths = outputPaths {
            try module.write(toFile: outputPaths[i])
        }

        /// LLGen
        if Options.shouldCompile.wasSet {
            guard let target = Options.target.value else {
                error("No compute target [hpvm | nvptx | amdgpu | cpu] was selected")
            }
            let cgen = codeGenerator(for: target, from: module)
            cgen.emitIR()

            /// Emit LLVM IR
            if Options.shouldEmitLLVM.wasSet {
                print(cgen.textualIR)
            }

            /// Emit bitcode
            if Options.shouldEmitBitcode.wasSet {
                try cgen.writeBitcode(toFile: filePath.replacingFileExtension(with: "bc"))
            }
        }
    }

}

do { try main() }
catch { print(error) }
