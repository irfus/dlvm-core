//
//  LossFunction.swift
//  DLVM
//
//  Created by Richard Wei on 2/7/17.
//
//

//open class LossFunctionPass : BasicBlockExtensionPass {
//
//    public struct Result : PassResultProtocol {
//        public var changed: Bool = false
//        public var errorValues: [(NamedValue, NamedValue)] = []
//        public init() {}
//    }
//
//    public var body: BasicBlock
//    public static let extensionType: BasicBlock.ExtensionType = .backpropagation
//    
//    public required init(body: BasicBlock) {
//        self.body = body
//    }
//
//    open func run() -> Result {
//        /// Remove extension if any
//        body.removeExtension(ofType: LossFunctionPass.extensionType)
//
//        var result = Result()
//
//        for output in module.outputs where body.exportedOutputs.contains(output) {
//            result.changed = true
//            let refInput = Input(name: output.name + ".ref", type: output.type, shape: output.shape)
//            module.insert(refInput)
//            /// TODO
//        }
//
//        return result
//    }
//
//}
//
//class CrossEntropyPass : LossFunctionPass {
//
//    open override func run() -> LossFunctionPass.Result {
//        let result = super.run()
//
//        return result
//    }
//    
//}
