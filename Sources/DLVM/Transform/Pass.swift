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

public protocol Pass : class {
    associatedtype Result : PassResultProtocol
    associatedtype Body : AnyObject
    var body: Body { get }
    func run() -> Result
}

public extension Pass where Body : IRObject, Body.Parent == Module {
    var module: Module? {
        return body.parent
    }

    func makeBuilder() -> IRBuilder? {
        return module.flatMap { IRBuilder(module: $0) }
    }
}

public extension Pass where Body : IRObject, Body.Parent == Function {
    var module: Module? {
        return body.parent?.parent
    }

    func makeBuilder() -> IRBuilder? {
        return module.flatMap { IRBuilder(module: $0) }
    }
}

public extension Pass where Body == BasicBlock {
    func makeBuilder() -> IRBuilder? {
        return module.flatMap {
            let builder = IRBuilder(module: $0)
            builder.move(to: body)
            return builder
        }
    }
}

public extension Pass where Body : IRObject, Body.Parent == BasicBlock {
    var module: Module? {
        return body.parent?.parent?.parent
    }

    func makeBuilder() -> IRBuilder? {
        return module.flatMap {
            let builder = IRBuilder(module: $0)
            builder.move(to: body.parent)
            return builder
        }
    }
}
