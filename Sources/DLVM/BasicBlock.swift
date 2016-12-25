//
//  BasicBlock.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

public class BasicBlock : IRCollection, IRObject {
    public var name: String
    public var elements: [Instruction]
    public weak var parent: Module?

    public init(name: String) {
        self.name = name
        elements = []
    }

    public init(name: String, instructions: [Instruction]) {
        self.name = name
        self.elements = instructions
    }
}
