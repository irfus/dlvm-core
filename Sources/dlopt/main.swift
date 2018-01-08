//
//  main.swift
//  dlopt
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

import DLVM
import DLParse
import DLCommandLineTools
import Foundation
import Basic
import Utility

class OptimizerToolOptions : ToolOptions {
    /// Bypass verification
    var noVerify = false
}

class OptimizerTool : CommandLineTool<OptimizerToolOptions> {
    public convenience init(args: [String]) {
        self.init(
            toolName: "dlopt",
            usage: "<inputs> [options]",
            overview: "DLVM IR optimizer",
            args: args
        )
    }

    override func runImpl() throws {
        let outputPaths = options.outputPaths
        if let outputPaths = outputPaths {
            guard outputPaths.count == options.inputFiles.count else {
                throw DLError.inputOutputCountMismatch
            }
        }

        /// Verify input files
        // NOTE: To be removed when PathArgument init checks for invalid paths.
        // Error should indicate raw string argument, not the corresponding path.
        if let invalidFile = options.inputFiles.first(where: { !isFile($0) }) {
            throw DLError.invalidInputFile(invalidFile)
        }

        for (i, inputFile) in options.inputFiles.enumerated() {
            /// Read IR and verify
            print("Source file:", inputFile.prettyPath())
            /// Parse
            let module = try Module.parsed(fromFile: inputFile.asString)

            /// Run passes
            if let passes = options.passes {
                for pass in passes {
                    try runPass(pass, on: module,
                                bypassingVerification: options.noVerify)
                }
            }

            /// Print IR instead of writing to file if requested
            if options.shouldPrintIR {
                print()
                print(module)
            }

            /// Otherwise, write result to IR file by default
            else {
                let path = outputPaths?[i] ?? inputFile
                try module.write(toFile: path.asString)
            }
        }
    }

    override class func defineArguments(parser: ArgumentParser, binder: ArgumentBinder<OptimizerToolOptions>) {
        binder.bind(
            option: parser.add(option: "--no-verify", kind: Bool.self,
                       usage: "Bypass verification after applying transforms"),
            to: { $0.noVerify = $1 })
    }
}

let tool = OptimizerTool(args: Array(CommandLine.arguments.dropFirst()))
tool.run()
