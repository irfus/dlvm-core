//
//  Users.swift
//  DLVM
//
//  Created by Richard Wei on 3/3/17.
//
//

public struct UserGraph {
    fileprivate var users: [ObjectIdentifier : ObjectSet<Instruction>] = [:]
}

internal extension UserGraph {

    mutating func insert(_ user: Instruction, for def: Definition) {
        let key = ObjectIdentifier(def)
        if !users.keys.contains(key) {
            users[key] = []
        }
        users[key]!.insert(user)
    }
    
}

public extension UserGraph {

    func users(of value: Definition) -> ObjectSet<Instruction> {
        return users[ObjectIdentifier(value)] ?? []
    }

    func users(of inst: Instruction) -> ObjectSet<Instruction> {
        guard let def = inst.definition else { return [] }
        return users(of: def)
    }

}

open class UserAnalysis : AnalysisPass<Function, UserGraph> {

    open override class func run(on body: Function) throws -> UserGraph {
        var userGraph = UserGraph()
        for inst in body.instructions {
            for use in inst.operands {
                if let def = use.definition {
                    userGraph.insert(inst, for: def)
                }
            }
        }
        return userGraph
    }

}
