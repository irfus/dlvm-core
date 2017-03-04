//
//  Graph.swift
//  DLVM
//
//  Created by Richard Wei on 2/19/17.
//
//

public protocol ForwardGraphNode {
    associatedtype SuccessorCollection: Collection // where SuccessorCollection.Iterator.Element == Self
    var successors: SuccessorCollection { get }
}

public protocol BackwardGraphNode {
    associatedtype PredecessorCollection: Collection
    var predecessors: PredecessorCollection { get }
}

public extension ForwardGraphNode {
    var isLeaf: Bool {
        return successors.isEmpty
    }
}

public extension BackwardGraphNode {
    var isSource: Bool {
        return predecessors.isEmpty
    }
}

public typealias BidirectionalGraphNode = ForwardGraphNode & BackwardGraphNode

public extension ForwardGraphNode
    where Self : BackwardGraphNode,
          SuccessorCollection.Iterator.Element == Self,
          Self.PredecessorCollection.Iterator.Element == Self {
    func reversedGraphNode() -> ReversedGraphNode<Self> {
        return ReversedGraphNode(base: self)
    }
}

public struct ReversedGraphNode<Base : BidirectionalGraphNode> : BidirectionalGraphNode
    where Base.SuccessorCollection.Iterator.Element == Base,
          Base.PredecessorCollection.Iterator.Element == Base {
    public let base: Base

    public init(base: Base) {
        self.base = base
    }

    public var successors: [ReversedGraphNode<Base>] {
        return base.predecessors.map{$0.reversedGraphNode()}
    }

    public var predecessors: [ReversedGraphNode<Base>] {
        return base.successors.map{$0.reversedGraphNode()}
    }
}


public protocol BidirectionalEdgeSet {
    associatedtype Node : AnyObject
    func predecessors(of node: Node) -> ObjectSet<Node>
    func successors(of node: Node) -> ObjectSet<Node>
}

extension BidirectionalEdgeSet {
    public var transpose: TransposeEdgeSet<Self> {
        return TransposeEdgeSet(base: self)
    }
}

public struct TransposeEdgeSet<Base : BidirectionalEdgeSet> : BidirectionalEdgeSet {

    public typealias Node = Base.Node

    public let base: Base

    public func predecessors(of node: Node) -> ObjectSet<Node> {
        return base.successors(of: node)
    }

    public func successors(of node: Node) -> ObjectSet<Node> {
        return base.predecessors(of: node)
    }
    
}

public struct DirectedGraph<Node : IRUnit> : BidirectionalEdgeSet {
    public struct Entry {
        public var successors: ObjectSet<Node> = []
        public var predecessors: ObjectSet<Node> = []
        fileprivate init() {}
    }

    fileprivate var entries: [Node : Entry] = [:]
}

// MARK: - Mutation
public extension DirectedGraph {
    mutating func insertNode(_ node: Node) {
        if contains(node) { return }
        entries[node] = Entry()
    }

    mutating func insertEdge(from src: Node, to dest: Node) {
        entries[src]?.successors.insert(dest)
        entries[dest]?.predecessors.insert(src)
    }

    mutating func removeAll() {
        entries.removeAll()
    }
}

// MARK: - Query
public extension DirectedGraph {

    func predecessors(of node: Node) -> ObjectSet<Node> {
        return self[node].predecessors
    }

    func successors(of node: Node) -> ObjectSet<Node> {
        return self[node].successors
    }

    subscript(node: Node) -> Entry {
        guard let entry = entries[node] else {
            preconditionFailure("Node not in the graph")
        }
        return entry
    }

    func contains(_ node: Node) -> Bool {
        return entries.keys.contains(node)
    }

    func containsEdge(from src: Node, to dest: Node) -> Bool {
        return successors(of: src).contains(dest)
    }

    func immediatelyPrecedes(_ firstNode: Node, _ secondNode: Node) -> Bool {
        return predecessors(of: secondNode).contains(firstNode)
    }

    func immediatelySucceeds(_ firstNode: Node, _ secondNode: Node) -> Bool {
        return successors(of: secondNode).contains(secondNode)
    }

    func precedes(_ firstNode: Node, _ secondNode: Node) -> Bool {
        guard contains(firstNode) else { return false }
        let secondPreds = predecessors(of: secondNode)
        return secondPreds.contains(firstNode)
            || secondPreds.contains(where: { precedes($0, secondNode) })
    }

    func succeeds(_ firstNode: Node, _ secondNode: Node) -> Bool {
        guard contains(secondNode) else { return false }
        let secondSuccs = successors(of: secondNode)
        return secondSuccs.contains(firstNode)
            || secondSuccs.contains(where: { succeeds(firstNode, $0) })
    }

    func isSource(_ node: Node) -> Bool {
        return predecessors(of: node).isEmpty
    }

    func isLeaf(_ node: Node) -> Bool {
        return successors(of: node).isEmpty
    }
    
}

// MARK: - Initializer from source nodes that store successors
public extension DirectedGraph
    where Node : ForwardGraphNode,
          Node.SuccessorCollection.Iterator.Element == Node {

    init<S : Sequence>(nodes: S) where S.Iterator.Element == Node {
        for node in nodes {
            insertNode(node)
            for succ in node.successors {
                insertEdge(from: node, to: succ)
            }
        }
    }
    
    init<S : Sequence>(sources: S) where S.Iterator.Element == Node {
        for source in sources {
            insertAll(fromSource: source)
        }
    }

    mutating func insertAll(fromSource node: Node) {
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
public extension DirectedGraph
    where Node : BackwardGraphNode,
          Node.PredecessorCollection.Iterator.Element == Node {

    init<S : Sequence>(nodes: S) where S.Iterator.Element == Node {
        for node in nodes {
            insertNode(node)
            for pred in node.predecessors {
                insertEdge(from: pred, to: node)
            }
        }
    }

    init<S : Sequence>(leaves: S) where S.Iterator.Element == Node {
        for leaf in leaves {
            insertAll(fromLeaf: leaf)
        }
    }

    mutating func insertAll(fromLeaf node: Node) {
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
