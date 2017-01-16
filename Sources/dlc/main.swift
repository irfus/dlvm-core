import Foundation

do {
    guard let file = CommandLine.arguments.dropFirst().first else {
        print("No file given")
        exit(0)
    }
    let telSource = try String(contentsOfFile: file, encoding: .utf8)
    print("Source file:", file)
    let ast = try ModuleNode.parser.parse(telSource)
    print("Parse\n", ast)
}
catch {
    print(error)
}
