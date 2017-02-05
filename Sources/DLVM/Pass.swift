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

public protocol Pass : SelfVerifiable {
    associatedtype Result : PassResultProtocol
    associatedtype Body : AnyObject
    var body: Body { get }
    func run() -> Result
    init(body: Body)
}

public protocol ExtensionGenerationPass : Pass {
    typealias Body = BasicBlock
    static var extensionType: BasicBlock.ExtensionType { get }
}

public extension ExtensionGenerationPass where Body : Global {
    public func verify() throws {
        guard let module = body.module else {
            throw VerificationError.globalMissingParent(body)
        }
        module.updateAnalysisInformation()
        try module.verify()
    }
}
