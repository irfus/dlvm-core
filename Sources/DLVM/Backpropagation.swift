//
//  Backpropagation.swift
//  DLVM
//
//  Created by Richard Wei on 2/7/17.
//
//

open class BackpropagationPass<Loss, Opt, AutoDiff> : BasicBlockExtensionPass
    where Loss : LossFunctionPass,
          Opt : OptimizerPass,
          AutoDiff : AutomaticDifferentiationPass
{

    public var body: BasicBlock

    public static var extensionType: BasicBlock.ExtensionType {
        return .backpropagation
    }

    public required init(body: BasicBlock) {
        self.body = body
    }

    public func run() -> PassResult {
        return PassResult()
    }
    
}
