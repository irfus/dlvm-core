//
//  Users.swift
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

public struct UserInfo {
    fileprivate var users: [ObjectIdentifier : Set<Instruction>] = [:]
}

internal extension UserInfo {
    mutating func insert(_ user: Instruction, for def: Definition) {
        let key = ObjectIdentifier(def)
        if !users.keys.contains(key) {
            users[key] = []
        }
        users[key]!.insert(user)
    }
}

// MARK: - Getter
public extension UserInfo {
    /// Returns a set of users
    subscript(value: Definition) -> Set<Instruction> {
        return users[ObjectIdentifier(value)] ?? []
    }
}

/// Analyzes function and produces a graph from definitions to users
open class UserAnalysis : AnalysisPass {
    public typealias Body = Function

    open class func run(on body: Function) throws -> UserInfo {
        var userGraph = UserInfo()
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
