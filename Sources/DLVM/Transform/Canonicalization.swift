//
//  Canonicalization.swift
//  DLVM
//
//  Created by Richard Wei on 4/4/17.
//
//

import Foundation

public class Canonicalization : TransformPass<Module> {
    public static override func run(on body: Module) throws -> Bool {
        let changed = try body.applyTransforms(GradientExpansion.self)
        body.stage = .canonical
        return changed
    }
}
