import TEL
import Foundation
import DLVM
//import DLVMReader
import DLVMCodeGen
import LLVM
import CommandLineKit

let cli = CommandLineKit.CommandLine()

enum Target : String {
    case nvvm = "nvvm"
    case hpvm = "hpvm"
}

struct Options {
    /// File
    static let filePaths = MultiStringOption(shortFlag: "f", longFlag: "files",
                                             helpMessage: "Paths to TEL source files")
    /// Output
    static let outputPaths = MultiStringOption(shortFlag: "o", longFlag: "outputs",
                                               helpMessage: "Output file paths")
    /// Target compute backend
    static let target = EnumOption<Target>(shortFlag: "t", longFlag: "target", helpMessage: "Target compute backend [ hpvm | nvvm ]")
    /// Emit DLVM IR
    static let shouldEmitDLVM = BoolOption(longFlag: "emit-dlvm", helpMessage: "Emit DLVM IR")
    /// Emit LLVM IR
    static let shouldEmitLLVM = BoolOption(longFlag: "emit-llvm", helpMessage: "Emit LLVM Textual IR")
    /// Emit LLVM Bitcode
    static let shouldEmitBitcode = BoolOption(longFlag: "emit-bc", helpMessage: "Emit LLVM Bitcode")
    /// Help
    static let needsHelp = BoolOption(shortFlag: "h", longFlag: "help",
                                      helpMessage: "Print help message")
}

cli.addOptions(Options.filePaths,
               Options.outputPaths,
               Options.target,
               Options.shouldEmitDLVM,
               Options.shouldEmitLLVM,
               Options.shouldEmitBitcode,
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
        let source = try String(contentsOfFile: filePath, encoding: .utf8)
        print("Source file:", filePath)
        let ast = try ProgramTree.parser.parse(source)
        let program = try Program(parse: ast)
        print("TEL module \"\(program.moduleName)\"")
        
        /// Generate IR
        let module = program.makeModule()

        /// Verify
        try module.verify()
        
        /// Run gradient expansion
        //try module.applyTransform(GradientExpander.self)

        /// Print DLVM IR if needed
        if Options.shouldEmitDLVM.wasSet {
            print("== DLVM module ==")
            print(module)
            /// Write IR
            try module.write(toFile: outputPath)
            print("DLVM module \"\(module.name)\" written to \(outputPath)")
        }

        /// LLGen
        guard let target = Options.target.value else { return }
            
        let llModule: LLVM.Module
        switch target {
        case .hpvm:
            let codeGen = CodeGenerator<HPVM>(module: module)
            llModule = codeGen.emit()
        case .nvvm:
            let codeGen = CodeGenerator<NVVM>(module: module)
            llModule = codeGen.emit()
        }

        /// Print LLVM IR if needed
        if Options.shouldEmitLLVM.wasSet {
            print("== LLVM module ==")
            llModule.dump()
        }
    }

}

do { try main() }
catch { print(error) }
