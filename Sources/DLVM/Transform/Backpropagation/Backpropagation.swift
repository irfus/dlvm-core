//
//  Backpropagation.swift
//  DLVM
//
//  Created by Richard Wei on 2/7/17.
//

open class BackpropagationPass : Pass {

    open let body: Function

    public required init(body: Function) {
        self.body = body
    }

    open func run() -> PassResult {
        var result = Result()
        guard let module = module else { return result }
        let builder = IRBuilder(module: module)

        /// Generate reference placeholders
        for ph in module.globals {
            if case let .placeholder(def) = ph {
                builder.declare(def.value, name: def.name + ".ref")
                result.changed = true
                /// Generate
            }
        }

        return result
    }
    
}
