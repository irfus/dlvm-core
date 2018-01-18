//
//  TransformUtilities.swift
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

// MARK: - Fresh name generators
fileprivate extension BasicBlock {
    var definedNames: Set<String> {
        return Set([name])
            .union(arguments.map { $0.name })
            .union(elements.flatMap { $0.name })
    }
}

fileprivate extension Function {
    var definedNames: Set<String> {
        return Set(elements.flatMap { $0.definedNames })
    }
}

internal func makeFreshName(_ name: String, in function: Function) -> String {
    var result = name
    var count = 0
    while function.definedNames.contains(result) {
        result = "\(name)_\(count)"
        count += 1
    }
    return result
}

internal func makeFreshBasicBlockName(_ name: String, in function: Function) -> String {
    var result = name
    var count = 0
    while function.elements.contains(where: { $0.name == result }) {
        result = "\(name)_\(count)"
        count += 1
    }
    return result
}

internal func makeFreshFunctionName(_ name: String, in module: Module) -> String {
    var result = name
    var count = 0
    while module.elements.map({ $0.name }).contains(result) {
        result = "\(name)_\(count)"
        count += 1
    }
    return result
}

internal extension Function {
    func makeFreshBasicBlockName(_ name: String) -> String {
        return DLVM.makeFreshBasicBlockName(name, in: self)
    }
}

internal extension BasicBlock {
    func makeFreshName(_ name: String) -> String {
        return DLVM.makeFreshName(name, in: parent)
    }

    func makeFreshBasicBlockName(_ name: String) -> String {
        return DLVM.makeFreshBasicBlockName(name, in: parent)
    }
}

// MARK: - Function cloning
public extension Function {
    /// Create clone of function
    public func makeClone(named name: String) -> Function {
        let newFunc = Function(name: name,
                               argumentTypes: argumentTypes,
                               returnType: returnType,
                               attributes: attributes,
                               declarationKind: declarationKind,
                               parent: parent)
        copyContents(to: newFunc)
        return newFunc
    }

    /// Copy basic blocks to an empty function
    public func copyContents(to other: Function) {
        /// Other function must be empty (has no basic blocks)
        guard other.isEmpty else {
            fatalError("""
                Could not copy contents to \(other.name) because it is not \
                empty
                """)
        }

        /// Mappings from old IR units to new IR units
        var newArgs: [Argument : Argument] = [:]
        var newBlocks: [BasicBlock : BasicBlock] = [:]
        var newInsts: [Instruction : Instruction] = [:]

        func newUse(from old: Use) -> Use {
            switch old {
            /// If recursion, replace function with new function
            case .function(_, self):
                return %other
            case .function, .variable:
                return old
            case let .literal(ty, lit) where lit.isAggregate:
                return .literal(
                    ty, lit.substituting(newUse(from: old), for: old))
            case let .literal(ty, lit):
                return .literal(ty, lit)
            case let .argument(_, arg):
                return %newArgs[arg]!
            case let .instruction(_, inst):
                return %newInsts[inst]!
            }
        }

        /// Clone basic blocks
        for oldBB in self {
            let newBB = BasicBlock(
                name: oldBB.name,
                arguments: oldBB.arguments.map{($0.name, $0.type)},
                parent: other)
            other.append(newBB)
            /// Insert arguments into mapping
            for (oldArg, newArg) in zip(oldBB.arguments, newBB.arguments) {
                newArgs[oldArg] = newArg
            }
            newBlocks[oldBB] = newBB
        }
        /// Clone instructions
        for oldBB in self {
            let newBB = newBlocks[oldBB]!
            for oldInst in oldBB {
                let newInst = Instruction(name: oldInst.name,
                                          kind: oldInst.kind, parent: newBB)
                /// Replace operands with new uses
                for oldUse in newInst.operands {
                    newInst.substitute(newUse(from: oldUse), for: oldUse)
                }
                /// If instruction branches, replace old BBs with new BBs
                switch newInst.kind {
                case let .branch(dest, args):
                    newInst.kind = .branch(newBlocks[dest]!, args)
                case let .conditional(cond, thenBB, thenArgs, elseBB, elseArgs):
                    newInst.kind = .conditional(
                        cond, newBlocks[thenBB]!, thenArgs,
                        newBlocks[elseBB]!, elseArgs)
                default: break
                }
                /// Insert instruction into mapping and new BB
                newInsts[oldInst] = newInst
                newBB.append(newInst)
            }
        }
    }
}

public extension BasicBlock {
    /// Creates a new basic block that unconditionally branches to self and
    /// hoists some predecessors to the new block.
    @discardableResult
    public func hoistPredecessorsToNewBlock<S : Sequence>(
        named name: String,
        hoisting predecessors: S) -> BasicBlock
        where S.Element == BasicBlock
    {
        let newBB = BasicBlock(
            name: makeFreshName(name),
            arguments: arguments.map{
                (makeFreshName($0.name), $0.type)
            },
            parent: parent)
        parent.insert(newBB, before: self)
        let builder = IRBuilder(basicBlock: newBB)
        builder.branch(self, newBB.arguments.map(%))
        /// Change all predecessors to branch to new block
        predecessors.forEach { pred in
            pred.terminator?.substituteBranches(to: self, with: newBB)
        }
        return newBB
    }
}
