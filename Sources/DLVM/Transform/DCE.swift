//
//  DCE.swift
//  DLVM
//
//  Created by Richard Wei on 4/26/17.
//
//

import Foundation

/// Dead code elimination (traditional algorithm)
open class DeadCodeElimination : TransformPass<Function> {
    open override static func run(on body: Function) throws -> Bool {
        var changed: Bool = false
        let workList: NSMutableOrderedSet = []
        var count = 0
        // Iterate over the original function, only adding instructions to the worklist
        // if they actually need to be revisited. This avoids having to pre-init
        // the worklist with the entire function's worth of instructions.
        for inst in body.instructions where !workList.contains(inst) {
            changed = try performDCE(on: inst, workList: workList, count: &count) || changed
        }
        while let inst = workList.lastObject as? Instruction {
            workList.remove(inst)
            changed = try performDCE(on: inst, workList: workList, count: &count) || changed
        }
        /// TODO: Print count when DEBUG
        return changed
    }

    private static func performDCE(on inst: Instruction,
                                   workList: NSMutableOrderedSet,
                                   count: inout Int) throws -> Bool {
        let function = inst.parent.parent
        let module = function.parent
        var userInfo = try function.analysis(from: UserAnalysis.self)
        let sideEffectInfo = try module.analysis(from: SideEffectAnalysis.self)

        /// If instruction is not trivially dead, change nothing
        guard userInfo[inst].isEmpty && sideEffectInfo[inst].mayHaveSideEffects else { return false }
        /// Eliminate
        inst.removeFromParent()
        /// Remove instruction and check users
        /// Get new user analysis
        userInfo = try function.analysis(from: UserAnalysis.self)
        /// For original uses, check if they need to be revisited
        for case let .instruction(_, usee) in inst.operands
            where userInfo[usee].isEmpty && sideEffectInfo[usee].mayHaveSideEffects {
            workList.add(usee)
        }
        return true
    }
}
