//
//  CFG.swift
//  DLVM
//
//  Created by Richard Wei on 2/24/17.
//

// MARK: - Basic block successors
public extension BasicBlock {

    var successorCount: Int {
        return terminator?.successorCount ?? 0
    }

    var hasSuccessors: Bool {
        return successorCount > 0
    }

}

// MARK: - Basic block graph traits
extension BasicBlock : BidirectionalGraphNode {
    public var successors: Set<BasicBlock> {
        guard let terminator = self.terminator else { return [] }
        switch terminator.kind {
        case let .control(.br(dest)):
            return [dest]
        case let .control(.condBr(_, thenBB, elseBB)):
            return [thenBB, elseBB]
        case let .operation(def):
            if case let .pull(_, thenBB, elseBB) = def.value {
                return [thenBB, elseBB]
            }
            fallthrough
        default:
            return []
        }
    }
}

// MARK: - Successors
public extension Control {
    var successors: [BasicBlock] {
        switch self {
        case let .br(bb): return [bb]
        case let .condBr(_, bb1, bb2): return [bb1, bb2]
        default: return []
        }
    }

    var successorCount: Int {
        switch self {
        case .br: return 1
        case .condBr: return 2
        default: return 0
        }
    }
}

// MARK: - Successors
public extension Operation {
    var successors: [BasicBlock] {
        switch self {
        case let .pull(_, bb1, bb2): return [bb1, bb2]
        default: return []
        }
    }

    var successorCount: Int {
        if case .pull = self { return 2 }
        else { return 0 }
    }
}

// MARK: - Instruction successors
public extension Instruction {

    var successors: [BasicBlock] {
        switch kind {
        case let .control(ctrl): return ctrl.successors
        case let .operation(def): return def.value.successors
        }
    }

    var successorCount: Int {
        switch kind {
        case let .control(ctrl): return ctrl.successorCount
        case let .operation(def): return def.value.successorCount
        }
    }

    var isCriticalEdge: Bool {
        if successorCount == 1 { return false }
        /// TODO: analyze precedessors
        return false
    }

}

// MARK: - Control flow property computation
public extension Function {

    /// Compute and returns back edges in function
    var backEdges: [(BasicBlock, BasicBlock)] {
        guard var bb = entry else { return [] }

        var visited: Set<BasicBlock> = []
        var visitStack: [BasicBlock] = []
        var inStack: Set<BasicBlock> = []
        var result: [(BasicBlock, BasicBlock)] = []

        /// Initialization
        visited.insert(bb)
        visitStack.append(bb)
        inStack.insert(bb)

        repeat {
            let parent = visitStack.removeFirst()
            var foundNew = false
            for succ in parent.successors {
                bb = succ
                visited.insert(bb)
                if bb.hasSuccessors {
                    foundNew = true
                    break
                }
                /// Successor is in visitStack, it's a back edge
                if inStack.contains(bb) {
                    result.append((parent, bb))
                }
            }
            if foundNew {
                inStack.insert(bb)
                visitStack.append(bb)
            } else {
                inStack.remove(visitStack.last!)
            }
        } while !visitStack.isEmpty

        return result
    }



}

// MARK: - Per-use analysis information
internal extension Use {
    func addUser(_ user: Instruction) {
        switch kind {
        case .literal: return
        case .global(let def): def.addUser(user)
        case .local(let def): def.addUser(user)
        case .argument(let def): def.addUser(user)
        }
    }
}

// MARK: - Per-instruction analysis information
internal extension Instruction {
    func updateUsers() {
        for use in operands {
            use.addUser(self)
        }
    }
}

// MARK: - Per-block analysis information
internal extension BasicBlock {

    /// Update user information
    private func updateUsers() {
        for inst in instructions {
            if case let .operation(oper) = inst.kind {
                oper.removeAllUsers()
            }
        }
        for inst in instructions {
            inst.updateUsers()
        }
    }

    /// Update predecessor of successors
    private func updateSuccessorPredecessors() {
        for succ in successors {
            succ.predecessors.removeAll()
        }
        for succ in successors {
            succ.predecessors.insert(self)
        }
    }

    /// Update analysis information
    func updateAnalysisInformation() {
        updateSuccessorPredecessors()
        updateUsers()
    }

}

// MARK: - Per-function analysis information
internal extension Function {

    func updateAnalysisInformation() {
        for bb in allBasicBlocks {
            bb.updateAnalysisInformation()
        }
    }

}
