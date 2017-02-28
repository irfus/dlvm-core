//
//  Backpropagation.swift
//  DLVM
//
//  Created by Richard Wei on 2/7/17.
//

public struct BackpropagationPass<E : ErrorFunctionBuilder, O : StochasticOptimizer> : Pass {

    public let body: Function

    public init(body: Function) {
        self.body = body
    }

    public func run() -> PassResult {
        var result = Result()
        guard let module = module, let forwardPass = forwardPass else { return result }
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
            let refOutPh = Placeholder(shape: outDef.shape,
                                                type: outDef.type,
                                                isRecurrent: outDef.isRecurrent)
            let refOutDef = builder.declare(refOutPh, name: outDef.name + ".ref")
            
            result.changed = true

            /// Generete a backward pass for each input (placeholder or function argument)
            for diffVar in forwardPass.differentiationVariables {
                /// Run error function pass
                let backwardPass = builder.buildBackwardPass(withRespectTo: diffVar, in: body)
                let backwardEntry = builder.buildBasicBlock(named: "entry", in: backwardPass)
                builder.move(to: backwardEntry)

                /// Emit get/pull in backward entry
                switch diffVar {
                case .argument(let arg):
                    fatalError("Unimplemented")

                case .global(let ph) where ph.isRecurrent:
                    fatalError("Recurrent placeholder is not supported yet")

                case .global(let ph):
                    let refVal = builder.buildOperation(.get(refOutDef))
                    let efGen = E(body: backwardEntry,
                                  output: outVal,
                                  referenceOutput: refVal)
                    _ = efGen.run()
                }
            }
        }

        return result
    }
    
}
