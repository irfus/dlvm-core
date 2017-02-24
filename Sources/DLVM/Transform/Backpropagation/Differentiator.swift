//
//  Differentiator.swift
//  DLVM
//
//  Created by Richard Wei on 2/21/17.
//
//

public protocol Differentiator : Pass {
    typealias Body = Function
    var root: Use { get }
    var variable: DifferentiationVariable { get }
    init(body: Function, root: Use, variable: DifferentiationVariable)
}

open class AutomaticDifferentiator : Differentiator {

    open let body: Function
    open let root: Use
    open let variable: DifferentiationVariable

    public required init(body: Function, root: Use, variable: DifferentiationVariable) {
        self.body = body
        self.root = root
        self.variable = variable
    }

    open func run() -> PassResult {
        var result = PassResult()
        guard let entry = body.entry else { return result }
        guard let builder = makeBuilder() else { return result }
        /// Search for differentiation variable from entry
        /// Return 0 if differentiation variable doesn't exist
        let allOperands = entry.preorder.lazy.flatMap({$0.elements}).flatMap({$0.operands})
        guard let foundVar = allOperands.first(where: { $0.definition === self.variable.definition }) else {
            let zero = variable.definition.makeZero()
            builder.buildControl(.ret(.literal(zero)))
            result.changed = true
            return result
        }

        /// Otherwise emit gradient
        /// TODO: differentiation!
        _ = foundVar

        result.changed = true
        return result
    }

}
