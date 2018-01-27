//
//  SideEffects.swift
//  DLVM
//
//  Copyright 2016-2018 The DLVM Team.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
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

public struct SideEffectInfo {
    private var table: [Function : SideEffectProperties] = [:]

    public subscript(body: Function) -> SideEffectProperties {
        set { table[body] = newValue }
        get { return table[body]! }
    }

    public subscript(instruction: Instruction) -> SideEffectProperties {
        switch instruction.kind {
        case .trap:
            return .mayTrap
        case _ where instruction.kind.mustWriteToMemory:
            return .mayWriteToMemory
        case .apply(.function(let callee), _):
            return self[callee]
        default:
            return []
        }
    }
}

/// Conservatively analyzes side effects of all functions in the module
open class SideEffectAnalysis : AnalysisPass {
    public typealias Body = Module
    public typealias Result = SideEffectInfo

    open class func run(on body: Module) -> SideEffectInfo {
        var result = SideEffectInfo()
        var sameModuleCalls: [(Function, Function)] = []

        /// Find instructions that definitely have side-effects, and collect
        /// callers and callees of functions
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
                case .apply(.function(let callee), _):
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
            for (caller, callee) in sameModuleCalls
                where result[caller] != result[callee] {
                result[caller].formUnion(result[callee])
                propChanged = true
            }
        } while propChanged

        return result
    }
}
