//
//  DFG.swift
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

public struct DataFlowGraph {
    fileprivate var users: [ObjectIdentifier : Set<Instruction>] = [:]
}

fileprivate extension DataFlowGraph {
    mutating func insert(_ user: Instruction, for def: Definition) {
        let key = ObjectIdentifier(def)
        if !users.keys.contains(key) {
            users[key] = []
        }
        users[key]!.insert(user)
    }
}

public extension DataFlowGraph {
    /// Returns a set of users
    func successors(of def: Definition) -> Set<Instruction> {
        return users[ObjectIdentifier(def)] ?? []
    }

    /// Returns a set of users within the basic block
    func successors(of def: Definition, in basicBlock: BasicBlock) -> Set<Instruction> {
        var users = successors(of: def)
        for user in users where user.parent != basicBlock {
            users.remove(user)
        }
        return users
    }

    /// Predecessors
    func predecessors(of inst: Instruction) -> [Definition] {
        return inst.operands.flatMap {$0.definition}
    }
}

public extension DataFlowGraph {
    func breadthFirst<C>(from definitions: C)
        -> AnyIterator<(depth: Int, user: Instruction)>
        where C : Collection, C.Iterator.Element == Definition
    {
        var depth = 1
        var queue: ArraySlice<Instruction?> = []
        var visited: Set<Instruction> = []
        var entryHasUsers = false
        for def in definitions {
            for user in successors(of: def) {
                entryHasUsers = true
                queue.append(user)
            }
        }
        if entryHasUsers {
            queue.append(nil)
        }
        return AnyIterator {
            while let first = queue.popFirst() {
                guard let this = first else {
                    depth += 1
                    continue
                }
                /// Skip visited
                guard !visited.contains(this) else {
                    continue
                }
                /// Push successors
                let successors = self.successors(of: this)
                for user in successors {
                    queue.append(user)
                }
                /// If successors were pushed, increment level
                if !successors.isEmpty {
                    queue.append(nil)
                }
                /// Mark visited
                visited.insert(this)
                return (depth: depth, user: this)
            }
            return nil
        }
    }
}

/// Analyzes function and produces a graph from definitions to users
open class DataFlowGraphAnalysis: AnalysisPass {
    public typealias Body = Function
    public typealias Result = DataFlowGraph

    open class func run(on body: Function) -> DataFlowGraph {
        var userGraph = DataFlowGraph()
        for inst in body.instructions {
            for use in inst.operands {
                if let def = use.value as? Definition {
                    userGraph.insert(inst, for: def)
                }
            }
        }
        return userGraph
    }
}

/// Convenience getter of users in Instruction
extension Instruction : ForwardGraphNode {
    public var users: Set<Instruction> {
        let fn = parent.parent
        let dfg = fn.analysis(from: DataFlowGraphAnalysis.self)
        return dfg.successors(of: self)
    }

    public var successors: Set<Instruction> {
        return users
    }
}

extension Instruction : BackwardGraphNode {
    public var predecessors: AnyCollection<Instruction> {
        return AnyCollection(operands.lazy.flatMap {
            $0.instruction
        })
    }
}
