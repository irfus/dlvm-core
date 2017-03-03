//
//  Differentiator.swift
//  DLVM
//
//  Created by Richard Wei on 2/21/17.
//
//

public class AutomaticDifferentiator : TransformationPass<Section> {

    open override class func run(on section: Section) -> Bool {
        var changed = false
        guard let entry = section.entry else { return false }
        let builder = IRBuilder(basicBlock: entry)

        return changed
    }

}
