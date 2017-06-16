import DLVM
import DLParse
import Foundation
import CommandLineKit
import DLVMCodeGen

let cli = CommandLineKit.CommandLine()

enum Target : String {
    case hpvm = "hpvm"
    case nvptx = "nvptx"
    case amdgpu = "amdgpu"
    case cpu = "cpu"
}

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

func error(_ message: String) -> Never {
    print("error: " + message)
    exit(EXIT_FAILURE)
}

func runPass(named name: String, on module: Module) throws {
    switch name {
    case "AD", "Differentiation":
        try module.applyTransform(Differentiation.self)
    case "Can", "Canonicalization":
        try module.applyTransform(Canonicalization.self)
    case "CP", "Checkpointing":
        try module.mapTransform(Checkpointing.self)
    case "DCE", "DeadCodeElimination":
        try module.mapTransform(DeadCodeElimination.self)
    case "CSE", "CommonSubexpressionElimination":
        try module.mapTransform(CommonSubexpressionElimination.self)
    case "AS", "AlgebraSimplification":
        try module.forEach { fn in try fn.applyTransform(AlgebraSimplification.self) }
    case "LAF", "LinearAlgebraFusion":
        try module.forEach { fn in try fn.mapTransform(LinearAlgebraFusion.self) }
    case "SP", "StackPromotion":
        try module.mapTransform(StackPromotion.self)
    case "VP", "ValuePromotion":
        try module.mapTransform(ValuePromotion.self)
    case "MCO", "MatrixChainOrdering":
        try module.forEach { fn in try fn.mapTransform(MatrixChainOrdering.self) }
    default:
        error("No transform pass named \(name)")
    }
}

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
