//
//  DFG.swift
//  DLVM
//
//  Created by Richard Wei on 3/2/17.
//
//

public struct BidirectionalGraph<Node : IRUnit> {
    public struct Entry : BidirectionalGraphNode {
        public var successors: ObjectSet<Node> = []
        public var predecessors: ObjectSet<Node> = []
        fileprivate init() {}
    }

    fileprivate var nodes: [Node : Entry] = [:]
}

// MARK: - Mutation
public extension BidirectionalGraph {
    mutating func insertNode(_ node: Node) {
        if contains(node) { return }
        nodes[node] = Entry()
    }

    mutating func insertEdge(from src: Node, to dest: Node) {
        nodes[src]?.successors.insert(dest)
        nodes[dest]?.predecessors.insert(src)
    }

    mutating func removeAll() {
        nodes.removeAll()
    }

}

// MARK: - Query
public extension BidirectionalGraph {

    public func predecessors(of node: Node) -> ObjectSet<Node>? {
        return self[node]?.predecessors
    }

    public func successors(of node: Node) -> ObjectSet<Node>? {
        return self[node]?.successors
    }

    public subscript(node: Node) -> Entry? {
        return nodes[node]
    }

    public func contains(_ node: Node) -> Bool {
        return nodes.keys.contains(node)
    }

    public func containsEdge(from src: Node, to dest: Node) -> Bool {
        return successors(of: src)?.contains(dest) ?? false
    }

    public func immediatelyPrecedes(_ firstNode: Node, _ secondNode: Node) -> Bool {
        return predecessors(of: secondNode)?.contains(firstNode) ?? false
    }

    public func immediatelySucceedes(_ firstNode: Node, _ secondNode: Node) -> Bool {
        return successors(of: secondNode)?.contains(secondNode) ?? false
    }

    public func predeces(_ firstNode: Node, _ secondNode: Node) -> Bool {
        guard contains(firstNode),
              let secondPreds = predecessors(of: secondNode) else {
            return false
        }
        return secondPreds.contains(firstNode)
            || secondPreds.contains(where: { predeces($0, secondNode) })
    }

    public func succeeds(_ firstNode: Node, _ secondNode: Node) -> Bool {
        guard contains(secondNode),
              let secondSuccs = successors(of: secondNode) else {
            return false
        }
        return secondSuccs.contains(firstNode)
            || secondSuccs.contains(where: { succeeds(firstNode, $0) })
    }
    
}

// MARK: - Initializer from source nodes that store successors
public extension BidirectionalGraph
    where Node : OutgoingGraphNode,
          Node.SuccessorCollection.Iterator.Element == Node {

    public init<S : Sequence>(nodes: S) where S.Iterator.Element == Node {
        for node in nodes {
            insertNode(node)
            for succ in node.successors {
                insertEdge(from: node, to: succ)
            }
        }
    }
    
    public init<S : Sequence>(sources: S) where S.Iterator.Element == Node {
        for source in sources {
            insertAll(fromSource: source)
        }
    }

    public mutating func insertAll(fromSource node: Node) {
        var visited: ObjectSet<Node> = []
        func insertAll(fromSource node: Node) {
            for succ in node.successors where !visited.contains(succ) {
                visited.insert(succ)
                insertNode(succ)
                insertEdge(from: node, to: succ)
                insertAll(fromSource: succ)
            }
        }
        insertAll(fromSource: node)
    }
}

// MARK: - Initializer from leaves that store predecessors
public extension BidirectionalGraph
    where Node : IncomingGraphNode,
          Node.PredecessorCollection.Iterator.Element == Node {

    public init<S : Sequence>(nodes: S) where S.Iterator.Element == Node {
        for node in nodes {
            insertNode(node)
            for pred in node.predecessors {
                insertEdge(from: pred, to: node)
            }
        }
    }

    public init<S : Sequence>(leaves: S) where S.Iterator.Element == Node {
        for leaf in leaves {
            insertAll(fromLeaf: leaf)
        }
    }

    public mutating func insertAll(fromLeaf node: Node) {
        var visited: ObjectSet<Node> = []
        func insertAll(fromLeaf node: Node) {
            for pred in node.predecessors {
                visited.insert(pred)
                insertNode(pred)
                insertEdge(from: node, to: node)
                insertAll(fromLeaf: pred)
            }
        }
        insertAll(fromLeaf: node)
    }

}

open class SectionGraphAnalysis : AnalysisPass<Function, BidirectionalGraph<Section>> {
    open override func run(on body: Function) -> BidirectionalGraph<Section> {
        return BidirectionalGraph(nodes: body)
    }
}

open class ControlFlowGraphAnalysis : AnalysisPass<Function, BidirectionalGraph<BasicBlock>> {
    open override func run(on body: Function) -> BidirectionalGraph<BasicBlock> {
        fatalError("Unimplemented")
        /// TODO: implement
    }
}

open class InstructionGraphAnalysis : AnalysisPass<Function, BidirectionalGraph<Instruction>> {
    open override func run(on body: Function) -> BidirectionalGraph<Instruction> {
        fatalError("Unimplemented")
        /// TODO: implement
    }
}
