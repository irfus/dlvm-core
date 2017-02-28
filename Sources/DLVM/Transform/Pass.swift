//
//  Pass.swift
//  DLVM
//
//  Created by Richard Wei on 1/15/17.
//
//

public protocol PassResultProtocol {
    var changed: Bool { get }
    init()
}

public struct PassResult : PassResultProtocol {
    public var changed: Bool = false
    public init() {}
}

public protocol Pass {
    associatedtype Result : PassResultProtocol
    associatedtype Body : AnyObject
    var body: Body { get }
    func run() -> Result
}

public extension Pass where Body : IRUnit, Body == Function {
    var module: Module? {
        return body.parent
    }

    var forwardPass: Function.Section? {
        return body.forwardPass
    }

    func makeBuilder() -> IRBuilder? {
        return module.flatMap { IRBuilder(module: $0) }
    }
}

public extension Pass where Body : IRUnit, Body == Function.Section {
    var module: Module {
        return body.parent.parent
    }

    func makeBuilder() -> IRBuilder? {
        return IRBuilder(module: module)
    }
}

public extension Pass where Body == BasicBlock {
    func makeBuilder() -> IRBuilder? {
        return IRBuilder(basicBlock: body)
    }
}
