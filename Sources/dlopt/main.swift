//
//  main.swift
//  dlopt
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

import DLVM
import DLParse
import DLCommandLineTools
import Foundation
import Basic
import Utility

public class Options {
    /// Input files
    var inputFiles: [AbsolutePath] = []
    /// Transformation passes
    var passes: [String]?
    /// Output paths
    var outputPaths: [AbsolutePath]?
    /// Print IR
    var shouldPrintIR = true
    /// Bypass verification
    var noVerify = false
}

/// Create the parser
let parser = ArgumentParser(commandName: "dlopt", usage: "[options] <inputs>",
                            overview: "DLVM IR optimizer")
/// Create the binder
let binder = ArgumentBinder<Options>()

/// Bind options
binder.bindArray(
    positional: parser.add(positional: "input files", kind: [PathArgument].self,
                           usage: "DLVM IR input files"),
    to: { $0.inputFiles = $1.lazy.map({ $0.path }) })

binder.bindArray(
    parser.add(option: "--passes", shortName: "-p", kind: [String].self,
               usage: "Transform passes"),
    parser.add(option: "--outputs", shortName: "-o", kind: [PathArgument].self,
               usage: "Output paths"),
    to: {
        if !$1.isEmpty { $0.passes = $1 }
        if !$2.isEmpty { $0.outputPaths = $2.lazy.map({ $0.path }) }
    })

binder.bind(
    parser.add(option: "--print-ir", kind: Bool.self,
               usage: "Print IR after transformation instead of writing to file"),
    parser.add(option: "--no-verify", kind: Bool.self,
               usage: "Bypass verification after applying transforms"),
    to: {
        $0.shouldPrintIR = $1 ?? $0.shouldPrintIR
        $0.noVerify = $2 ?? $0.noVerify
    })

func main() throws {
    /// Parse arguments
    var options = Options()
    do {
        let arguments = Array(CommandLine.arguments.dropFirst())
        let result = try parser.parse(arguments)
        binder.fill(result, into: &options)
    } catch ArgumentParserError.expectedArguments(_, ["input files"]) {
        throw Error.noInputPaths
    }

    let outputPaths = options.outputPaths
    if let outputPaths = outputPaths {
        guard outputPaths.count == options.inputFiles.count else {
            throw Error.inputOutputCountMismatch
        }
    }

    /// Verify input files
    // NOTE: To be removed when PathArgument init checks for invalid paths.
    // Error should indicate raw string argument, not the corresponding path.
    if let invalidPath = options.inputFiles.first(where: { !isFile($0) }) {
        throw Error.invalidInputPath(invalidPath)
    }

    for (i, inputFile) in options.inputFiles.enumerated() {
        /// Read IR and verify
        print("Source file:", inputFile.prettyPath())
        /// Parse
        let module = try Module.parsed(fromFile: inputFile.asString)

        /// Run passes
        if let passes = options.passes {
            for passName in passes {
                try runPass(named: passName, on: module,
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

private enum Error: Swift.Error {
    /// No input paths were specified.
    case noInputPaths

    /// An input path is invalid.
    case invalidInputPath(AbsolutePath)

    /// The number of input files and output paths do not match.
    case inputOutputCountMismatch
}

extension Error: CustomStringConvertible {
    var description: String {
        switch self {
        case .noInputPaths:
            return "no input files"
        case .invalidInputPath(let path):
            return "invalid input path: \(path.prettyPath())"
        case .inputOutputCountMismatch:
            return "number of inputs and outputs do not match"
        }
    }
}

do { try main() }
catch let err { error(err) }
