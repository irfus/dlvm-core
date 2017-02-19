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

public protocol Pass : class, SelfVerifiable {
    associatedtype Result : PassResultProtocol
    associatedtype Body : AnyObject
    var body: Body { get }
    func run() -> Result
    init(body: Body)
}
