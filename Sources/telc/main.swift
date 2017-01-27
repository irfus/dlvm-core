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
    let program = try Program(parse: ast)
    let module = program.makeModule()
    // Write to file
    let url = URL(fileURLWithPath: file)
    let dlUrl = url.deletingPathExtension().appendingPathExtension("dl")
    var text = ""
    module.write(to: &text)
    try text.write(to: dlUrl, atomically: true, encoding: .utf8)
}
catch {
    print(error)
}
