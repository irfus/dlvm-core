//
//  Pass.swift
//  DLVM
//
//  Created by Richard Wei on 1/15/17.
//
//

//public protocol PassResultProtocol {
//    var changed: Bool { get }
//    init()
//}
//
//public struct PassResult : PassResultProtocol {
//    public var changed: Bool = false
//    public init() {}
//}
//
//public protocol Pass : class, SelfVerifiable {
//    associatedtype Result : PassResultProtocol
//    associatedtype Body : AnyObject
//    var body: Body { get }
//    func run() -> Result
//    init(body: Body)
//}
//
//public extension Pass where Body : Global {
//    var module : Module {
//        guard let module = body.module else {
//            preconditionFailure("Module not available")
//        }
//        return module
//    }
//}
//
//public protocol BasicBlockPass : Pass {
//    typealias Body = BasicBlock
//}
//
//public protocol BasicBlockExtensionPass : BasicBlockPass {
//    static var extensionType: BasicBlock.ExtensionType { get }
//}
//
//public extension Pass where Body : Global {
//    public func verify() throws {
//        guard let module = body.module else {
//            throw VerificationError.globalMissingParent(body)
//        }
//        try module.verify()
//    }
//}
