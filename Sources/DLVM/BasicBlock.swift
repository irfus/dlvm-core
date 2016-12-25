//
//  BasicBlock.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

public class BasicBlock {
    public var name: String
    private var instructions: [Instruction] = []

    public init(name: String) {
        self.name = name
    }
}
