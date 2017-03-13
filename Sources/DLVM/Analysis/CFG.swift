//
//  CFG.swift
//  DLVM
//
//  Created by Richard Wei on 2/24/17.
//

// MARK: - Basic block graph traits
extension BasicBlock : ForwardGraphNode {
    public var successors: ObjectSet<BasicBlock> {
        guard let terminator = self.terminator else { return [] }
        switch terminator.kind {
        case let .control(.branch(dest, _)):
            return [dest]
        case let .control(.conditional(_, thenBB, elseBB)),
             let .control(.pull(_, thenBB, elseBB)):
            return [thenBB, elseBB]
        default:
            return []
        }
    }
}

// MARK: - Successors
public extension Control {
    var successors: ObjectSet<BasicBlock> {
        /// TODO: Include entries of backward passes as successors
        switch self {
        case let .branch(bb, _): return [bb]
        case let .conditional(_, bb1, bb2): return [bb1, bb2]
        default: return []
        }
    }

    var successorCount: Int {
        switch self {
        case .branch: return 1
        case .conditional, .pull: return 2
        default: return 0
        }
    }
}

// MARK: - Instruction successors
public extension Instruction {

    var successors: ObjectSet<BasicBlock> {
        switch kind {
        case let .control(ctrl): return ctrl.successors
        default: return []
        }
    }

    var successorCount: Int {
        switch kind {
        case let .control(ctrl): return ctrl.successorCount
        default: return 0
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
        var bb = entry
        var visited: ObjectSet<BasicBlock> = []
        var visitStack: [BasicBlock] = []
        var inStack: ObjectSet<BasicBlock> = []
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
