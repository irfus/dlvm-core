//
//  Differentiator.swift
//  DLVM
//
//  Created by Richard Wei on 2/21/17.
//
//

public class AutomaticDifferentiator : TransformationPass<Section> {

    public let root: Use
    public let variable: DifferentiationVariable

    public init(root: Use, variable: DifferentiationVariable) {
        self.root = root
        self.variable = variable
    }

    open override func run(on section: Section) -> Bool {
        var changed = false
        guard let entry = section.entry else { return false }
        let builder = IRBuilder(basicBlock: entry)

        /// Search for differentiation variable from entry
        /// Return 0 if differentiation variable doesn't exist
        let allOperands = entry.preorder.lazy.flatMap({$0.elements}).flatMap({$0.operands})
        guard let foundVar = allOperands.first(where: { $0.definition === self.variable.definition }) else {
            let zero = variable.definition.makeZero()
            builder.buildControl(.ret(.literal(zero)))
            changed = true
            return changed
        }

        /// Otherwise emit gradient
        /// TODO: differentiation!
        _ = foundVar

        return changed
    }

}

