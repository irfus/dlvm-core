//
//  Backpropagation.swift
//  DLVM
//
//  Created by Richard Wei on 2/7/17.
//

public class BackpropagationPass : TransformationPass<Function> {

    public typealias Body = Function

    @discardableResult
    open override func run(on function: Function) -> Bool {
        let module = function.parent
        var changed = false
        guard let forwardPass = function.forwardPass else { return false }
        let builder = IRBuilder(module: module)
        /// Generate reference placeholders
        for global in module.globals {
            guard case let .output(outDef) = global,
                let endBlock: BasicBlock = forwardPass.endBlock,
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
            
            /// Generete a backward pass for each input (placeholder or function argument)
            for diffVar in forwardPass.differentiationVariables {
                /// Run error function pass
                let backwardPass = builder.buildSection(named: "backward", dependingOn: [forwardPass], in: function)
                
                let backwardEntry = builder.buildBasicBlock(named: "entry", in: backwardPass)
                builder.move(to: backwardEntry)

                /// Emit get/pull in backward entry
                switch diffVar {
                case .argument(let arg):
                    fatalError("Unimplemented")

                case .global(let ph) where ph.isRecurrent:
                    fatalError("Recurrent placeholder is not supported yet")

                case .global(let ph):
                    let _ = builder.buildOperation(.get(refOutDef))
                }
            }
        }

        return changed
    }
    
}
