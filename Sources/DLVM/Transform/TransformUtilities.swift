//
//  TransformUtilities.swift
//  DLVM
//
//  Copyright 2016-2017 Richard Wei.
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

// MARK: - Function cloning
/// - Note: Big, ugly, not-so-safe, imperative code written in 4 minutes
public extension Function {
    public func makeClone(named name: String) -> Function {
        let newFunc = Function(name: name,
                               argumentTypes: argumentTypes,
                               returnType: returnType,
                               attributes: attributes,
                               declarationKind: declarationKind,
                               parent: parent)

        /// Mappings from old IR units to new IR units
        var newArgs: [Argument : Argument] = [:]
        var newBlocks: [BasicBlock : BasicBlock] = [:]
        var newInsts: [Instruction : Instruction] = [:]

        func newUse(from old: Use) -> Use {
            switch old {
            /// If recursion, change function to the new function
            case .function(_, self):
                return %newFunc
            case .function, .global:
                return old
            case let .literal(ty, lit):
                return .literal(ty, lit.substituting(newUse(from: old), for: old))
            case let .argument(_, arg):
                return %newArgs[arg]!
            case let .instruction(_, inst):
                return %newInsts[inst]!
            case let .constant(_, instKind):
                return %instKind.substituting(newUse(from: old), for: old)
            }
        }

        /// Clone basic blocks
        for oldBB in self {
            let newBB = BasicBlock(name: oldBB.name,
                                   arguments: oldBB.arguments.map{($0.name, $0.type)},
                                   parent: newFunc)
            newFunc.append(newBB)

            /// Insert argument mappings
            for (oldArg, newArg) in zip(oldBB.arguments, newBB.arguments) {
                newArgs[oldArg] = newArg
            }
            newBlocks[oldBB] = newBB
        }

        /// Clone instructions
        for oldBB in self {
            let newBB = newBlocks[oldBB]!
            /// Clone instructions
            for oldInst in oldBB {
                let newInst = Instruction(name: oldInst.name, kind: oldInst.kind, parent: newBB)
                /// - Note: Slow but clean for now
                for oldUse in newInst.operands {
                    newInst.substitute(oldUse, for: newUse(from: oldUse))
                }
                /// If branching, switch old BBs to new BBs
                switch newInst.kind {
                case let .branch(dest, args):
                    newInst.kind = .branch(newBlocks[dest]!, args)
                case let .conditional(cond, thenBB, thenArgs, elseBB, elseArgs):
                    newInst.kind = .conditional(cond, newBlocks[thenBB]!, thenArgs,
                                                newBlocks[elseBB]!, elseArgs)
                default: break
                }
            }
        }

        return newFunc
    }
}
