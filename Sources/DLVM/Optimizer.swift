//
//  Optimizer.swift
//  DLVM
//
//  Created by Richard Wei on 2/7/17.
//
//

open class OptimizerPass : BasicBlockExtensionPass {

    public struct Result : PassResultProtocol {
        public var changed: Bool = false
        public var optimizer: [(NamedValue, NamedValue)] = []
        public init() {}
    }

    public var body: BasicBlock
    public static let extensionType: BasicBlock.ExtensionType = .backpropagation

    public required init(body: BasicBlock) {
        self.body = body
    }

    open func run() -> Result {
        /// Remove extension if any
        body.removeExtension(ofType: LossFunctionPass.extensionType)

        /// TODO

        return Result()
    }
    
}
