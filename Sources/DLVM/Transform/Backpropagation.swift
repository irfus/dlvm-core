//
//  Backpropagation.swift
//  DLVM
//
//  Created by Richard Wei on 2/7/17.
//
//
//
//open class BackpropagationPass<Loss, Opt, AutoDiff> : BasicBlockExtensionPass
//    where Loss : LossFunctionPass,
//          Opt : OptimizerPass,
//          AutoDiff : AutomaticDifferentiationPass
//{
//
//    public var body: BasicBlock
//
//    lazy var lossFunctionPass: Loss = Loss(body: self.body)
//    lazy var autoDiffPass: AutoDiff = AutoDiff(body: self.body)
//    lazy var optimizerPass: Opt = Opt(body: self.body)
//
//
//    public static var extensionType: BasicBlock.ExtensionType {
//        return .backpropagation
//    }
//
//    public required init(body: BasicBlock) {
//        self.body = body
//    }
//
//    public func run() -> PassResult {
//        return PassResult()
//    }
//    
//}
