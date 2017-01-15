//
//  Pass.swift
//  DLVM
//
//  Created by Richard Wei on 1/15/17.
//
//

public protocol Pass {
    var changed: Bool { get }
    mutating func run()
}

public protocol ModulePass : Pass {
    var module: Module { get }
    init(module: Module)
}

public protocol BasicBlockPass : Pass {
    var basicBlock: BasicBlock { get }
    init(basicBlock: BasicBlock)
}
