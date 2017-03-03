//
//  Backpropagation.swift
//  DLVM
//
//  Created by Richard Wei on 2/7/17.
//

public class BackpropagationPass : TransformationPass<Module> {

    public typealias Body = Function

    @discardableResult
    open override class func run(on module: Module) -> Bool {
        var changed = false
        let builder = IRBuilder(module: module)

        guard let main = module.mainFunction else { return false }
        guard let forward = main.top else { return false }

        /// Generate reference placeholder
        for global in module.globals {
            guard case let .output(outDef) = global,
                let endBlock: BasicBlock = main.top?.endBlock,
                case .control(.yield(let outVal, outDef))? = endBlock.terminator?.kind else {
                    // TODO: error handling
                    continue
            }
            
            /// Emit a reference output placeholder for error computation
            changed = true
            let refOutPh = Placeholder(shape: outDef.shape,
                                       type: outDef.type,
                                       isRecurrent: outDef.isRecurrent)
            let refOutDef = builder.declare(refOutPh, name: outDef.name + ".ref")
            
        }

        return changed
    }
    
}
