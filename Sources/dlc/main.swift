import DLVM
import DLParse
import Foundation
import CommandLineKit
import DLVMCodeGen

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
}

cli.addOptions(Options.filePaths,
               Options.shouldPrintIR,
               Options.passes,
               Options.outputPaths,
               Options.needsHelp)

/// Parse command line
do { try cli.parse(strict: true) }
catch { cli.printUsage(error); exit(EXIT_FAILURE) }

func error(_ message: String) {
    print("error: " + message)
    exit(EXIT_FAILURE)
}

func runPass(named name: String, on module: Module) throws {
    switch name {
    case "Canonicalization":
        try module.applyTransform(Canonicalization.self)
    case "DCE":
        try module.mapTransform(DeadCodeElimination.self)
    case "CSE":
        try module.mapTransform(CommonSubexpressionElimination.self)
    case "AD", "Differentiation":
        try module.applyTransform(Differentiation.self)
    case "AS", "AlgebraSimplification":
        try module.forEach { fn in try fn.mapTransform(AlgebraSimplification.self) }
    case "LAF", "LinearAlgebraFusion":
        try module.forEach { fn in try fn.mapTransform(LinearAlgebraFusion.self) }
    case "StackPromotion":
        try module.mapTransform(StackPromotion.self)
    case "ValuePromotion":
        try module.mapTransform(ValuePromotion.self)
    case "MCO", "MatrixChainOrdering":
        try module.forEach { fn in try fn.mapTransform(MatrixChainOrdering.self) }
    default:
        error("No transform pass named \(name)")
    }
}

func main() throws {
    
    guard !Options.needsHelp.wasSet else {
        print("DLVM IR compiler\n")
        cli.printUsage()
        return
    }

    guard let filePaths = Options.filePaths.value else {
        error("no input files; use -f to specify files")
        return
    }

    let outputPaths = Options.outputPaths.value
    if let outputPaths = outputPaths {
        guard outputPaths.count == filePaths.count else {
            error("different numbers of inputs and outputs specified")
            return
        }
    }

    for (i, filePath) in filePaths.enumerated() {
        /// Read IR and verify
        let irSource = try String(contentsOfFile: filePath, encoding: .utf8)
        print("Source file:", filePath)
        /// Lex and parse
        let parser = try Parser(text: irSource)
        let module = try parser.parseModule()
        try module.verify()

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
        if let outputPaths = outputPaths {
            /// Write IR
            try module.write(toFile: outputPaths[i])
        }
    }

}

do { try main() }
catch { print(error) }
