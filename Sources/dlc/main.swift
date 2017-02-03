import Foundation
import DLVMReader

do {
    guard let file = CommandLine.arguments.dropFirst().first else {
        print("No file given")
        exit(0)
    }
    let irSource = try String(contentsOfFile: file, encoding: .utf8)
    print("Source file:", file)
    let ast = try ModuleNode.parser.parse(irSource)
    print(ast.range)
    let module = try ast.makeModule()
    print(module)
}
catch {
    print(error)
}
