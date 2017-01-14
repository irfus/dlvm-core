import TEL
import Foundation

do {
    guard let file = CommandLine.arguments.dropFirst().first else {
        print("No file given")
        exit(0)
    }
    let telSource = try String(contentsOfFile: file, encoding: .utf8)
    print("Source file:", file)
    let ast = try ProgramTree.parser.parse(telSource)
    print("Parse\n", ast)
    let program = try Program(parse: ast)
    print("Type-checking passed")
    let module = program.makeModule()
    print(module)
}
catch {
    print(error)
}
