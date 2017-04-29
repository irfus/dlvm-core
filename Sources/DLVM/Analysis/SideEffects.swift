//
//  SideEffects.swift
//  DLVM
//
//  Created by Richard Wei on 4/26/17.
//
//

import Foundation

public struct SideEffectProperties : OptionSet {
    public let rawValue: Int
    public static let mayWriteToMemory = SideEffectProperties(rawValue: 1 << 0)
    public static let mayTrap = SideEffectProperties(rawValue: 1 << 1)
    public static let none: SideEffectProperties = []
    public static let all: SideEffectProperties = [.mayWriteToMemory, .mayTrap]

    public init(rawValue: Int) {
        self.rawValue = rawValue
    }
}

public struct SideEffectInfo<Unit : IRUnit> {
    private var table: [Unit : SideEffectProperties] = [:]

    public subscript(body: Unit) -> SideEffectProperties {
        set { table[body] = newValue }
        get { return table[body]! }
    }
}

/// Conservatively analyzes side effects of all functions in the module
public class FunctionSideEffectAnalysis : AnalysisPass<Module, SideEffectInfo<Function>> {
    public override static func run(on body: Module) throws -> SideEffectInfo<Function> {
        var result = SideEffectInfo<Function>()
        var sameModuleCalls: [(Function, Function)] = []

        /// Find instructions that definitely have side-effects, and collect callers and
        /// callees of functions
        for function in body {
            var props: SideEffectProperties = []
            for inst in function.instructions {
                /// Check instructions that definitely have side effects
                if inst.kind.mustWriteToMemory {
                    props.insert(.mayWriteToMemory)
                }
                if inst.kind.isTrap {
                    props.insert(.mayTrap)
                }
                /// Check function calls
                switch inst.kind {
                case .apply(.function(_, let callee), _):
                    /// Call within the same module
                    sameModuleCalls.append((function, callee))
                case .apply:
                    /// External call: we make conservative decision
                    props = .all
                default:
                    break
                }
            }
            result[function] = props
        }

        /// For each function call, union caller's properties with callee's
        var propChanged = false
        repeat {
            for (caller, callee) in sameModuleCalls where result[caller] != result[callee] {
                result[caller].formUnion(result[callee]) // Unwrapped safely
                propChanged = true
            }
        } while propChanged

        return result
    }
}

/// Conservatively analyzes side effects of all instructions in the module
/// - Dependencies: FunctionSideEffectAnalysis
public class SideEffectAnalysis : AnalysisPass<Module, SideEffectInfo<Instruction>> {
    public override static func run(on body: Module) throws -> SideEffectInfo<Instruction> {
        var funcSideEffects = try body.analysis(from: FunctionSideEffectAnalysis.self)
        var result = SideEffectInfo<Instruction>()
        for function in body {
            for inst in function.instructions {
                switch inst.kind {
                case _ where inst.kind.mustWriteToMemory:
                    result[inst].insert(.mayWriteToMemory)
                case _ where inst.kind.isTrap:
                    result[inst].insert(.mayTrap)
                case .apply(.function(_, let callee), _):
                    result[inst].formUnion(funcSideEffects[callee])
                default:
                    result[inst] = .none
                }
            }
        }
        return result
    }
}
