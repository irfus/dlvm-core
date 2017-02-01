//
//  Pass.swift
//  DLVM
//
//  Created by Richard Wei on 1/15/17.
//
//

public protocol PassResultProtocol {
    var changed: Bool { get }
}

public struct PassResult : PassResultProtocol {
    public var changed: Bool
}

public protocol Pass {
    associatedtype Result : PassResultProtocol
    associatedtype Body : AnyObject
    var body: Body { get }
    func run() -> Result
    init(body: BasicBlock)
}
