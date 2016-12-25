//
//  Module.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

/// Module representing a neural network
public class Module : IRCollection {
    var definitions: [Operand]
    public var elements: [BasicBlock]

    public init() {
        definitions = []
        elements = []
    }

    public init(definitions: [Operand], basicBlocks: [BasicBlock]) {
        self.definitions = definitions
        self.elements = basicBlocks
    }
}
