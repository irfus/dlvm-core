//
//  Backpropagation.swift
//  DLVM
//
//  Created by Richard Wei on 2/7/17.
//

public struct BackpropagationPass : Pass {

    public let body: Function

    public init(body: Function) {
        self.body = body
    }

    public func run() -> PassResult {
        var result = Result()
        guard let module = module else { return result }
        let builder = IRBuilder(module: module)

        /// Generate reference placeholders
        for global in module.globals {
            if case let .placeholder(def) = global, def.isUsed(in: body.forwardPass) {
                builder.declare(def.value, name: def.name + ".ref")
                result.changed = true
                /// Generate
            }
        }

        return result
    }
    
}
