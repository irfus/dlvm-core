import TEL
import Foundation

do {
    guard let file = CommandLine.arguments.dropFirst().first else {
        print("No file given")
        exit(0)
    }
//    guard let path = Bundle.main.path(forResource: file, ofType: nil) else {
//        print("Cannot open file \(file)")
//        exit(0)
//    }
    let telSource = try String(contentsOfFile: file, encoding: .utf8)
    let ast = try ProgramTree.parser.parse(telSource)
    dump(ast)
}
catch {
    print(error)
}
