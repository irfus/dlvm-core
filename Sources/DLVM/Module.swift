//
//  Module.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

/// Module representing a neural network
public class Module : IRCollection {
    var declarations: [Variable]
    public var elements: [BasicBlock]

    public init() {
        declarations = []
        elements = []
    }

    public init(declarations: [Variable], basicBlocks: [BasicBlock]) {
        self.declarations = declarations
        self.elements = basicBlocks
    }
}

public extension Module {

    public func declare(_ variable: Variable) {
        declarations.append(variable)
    }
    
}
