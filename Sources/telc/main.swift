import TEL
import Foundation
import DLVM
//import DLVMReader
import CommandLineKit

let cli = CommandLineKit.CommandLine()

struct Options {
    /// File
    static let filePaths = MultiStringOption(shortFlag: "f", longFlag: "files",
                                             helpMessage: "Paths to TEL source files")
    /// Output
    static let outputPaths = MultiStringOption(shortFlag: "o", longFlag: "outputs",
                                               helpMessage: "Output file paths")
    /// Print IR
    static let shouldPrintIR = BoolOption(longFlag: "print-ir",
                                          helpMessage: "Print DLVM IR after compilation")
    /// Help
    static let needsHelp = BoolOption(shortFlag: "h", longFlag: "help",
                                      helpMessage: "Print help message")
}

cli.addOptions(Options.filePaths,
               Options.outputPaths,
               Options.shouldPrintIR,
               Options.needsHelp)

/// Parse command line
do { try cli.parse(strict: true) }
catch { cli.printUsage(error); exit(EXIT_FAILURE) }

func error(_ message: String) {
    print("error: " + message)
    exit(EXIT_FAILURE)
}

func main() throws {

    guard !Options.needsHelp.wasSet else {
        print("Deep Learning Virtual Machine")
        print("Tensor Expression Language Compiler\n")
        cli.printUsage()
        return
    }

    guard let filePaths = Options.filePaths.value else {
        error("no input files; use -f to specify files")
        return
    }

    if let outputPaths = Options.outputPaths.value, outputPaths.count != filePaths.count {
        error("different numbers of inputs and outputs specified")
    }

    let outputPaths = Options.outputPaths.value ?? filePaths.map { filePath in
        let url = URL(fileURLWithPath: filePath)
        let dlUrl = url.deletingPathExtension().appendingPathExtension("dl")
        return dlUrl.relativePath
    }

    for (filePath, outputPath) in zip(filePaths, outputPaths) {
        let source = try String(contentsOfFile: filePath)
        print("Source file:", filePath)
        let ast = try ProgramTree.parser.parse(source)
        let program = try Program(parse: ast)
        print("TEL module \"\(program.moduleName)\"")
        /// Generate IR
        let module = program.makeModule()
        try module.verify()
        try module.applyTransform(GradientExpander.self)

        /// Print IR if needed
        if Options.shouldPrintIR.wasSet {
            print(module)
        }

        /// Write IR
        try module.write(toFile: outputPath)
        print("DLVM module \"\(module.name)\" written to \(outputPath)")
    }

}

do { try main() }
catch { print(error) }
