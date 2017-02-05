//
//  Reader.swift
//  DLVM
//
//  Created by Richard Wei on 2/3/17.
//
//

import class DLVM.Module
import struct Foundation.URL

public extension Module {

    public convenience init(parse: ModuleNode) throws {
        self.init(name: parse.name)

        for decl in parse.declarations {
            try decl.addDeclaration(to: self)
        }

        for bbNode in parse.basicBlocks {
            let bb = try bbNode.makeBasicBlock(in: nil, module: self)
            if !bb.isExtension {
                insert(bb)
            }
        }

        updateAnalysisInformation()
        try verify()
    }

    public convenience init(contentsOfFile path: String) throws {
        let url = URL(fileURLWithPath: path)
        let text = try String(contentsOf: url)
        let parse = try ModuleNode.parser.parse(text)
        try self.init(parse: parse)
    }
    
}
