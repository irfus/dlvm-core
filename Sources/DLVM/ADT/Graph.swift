//
//  Graph.swift
//  DLVM
//
//  Created by Richard Wei on 2/19/17.
//
//

/// Graph node that defines its successors
public protocol ForwardGraphNode {
    associatedtype SuccessorCollection: Collection // where SuccessorCollection.Iterator.Element == Self
    var successors: SuccessorCollection { get }
}

/// Graph node that defines its predecessors
public protocol BackwardGraphNode {
    associatedtype PredecessorCollection: Collection
    var predecessors: PredecessorCollection { get }
}

// MARK: - Property predicates
public extension ForwardGraphNode {
    var isLeaf: Bool {
        return successors.isEmpty
    }
}

// MARK: - Property predicates
public extension BackwardGraphNode {
    var isSource: Bool {
        return predecessors.isEmpty
    }
}

/// Graph node that defines both it's succssors and predecessors
public typealias BidirectionalGraphNode = ForwardGraphNode & BackwardGraphNode

/// Transpose of a bidirectional graph node
public struct TransposeGraphNode<Base : BidirectionalGraphNode> : BidirectionalGraphNode
    where Base.SuccessorCollection.Iterator.Element == Base,
          Base.PredecessorCollection.Iterator.Element == Base {
    public let base: Base

    public init(base: Base) {
        self.base = base
    }

    public var successors: [TransposeGraphNode<Base>] {
        return base.predecessors.map{$0.transpose}
    }

    public var predecessors: [TransposeGraphNode<Base>] {
        return base.successors.map{$0.transpose}
    }
}

// MARK: - Transpose
public extension ForwardGraphNode
    where Self : BackwardGraphNode,
          SuccessorCollection.Iterator.Element == Self,
          Self.PredecessorCollection.Iterator.Element == Self {
    var transpose: TransposeGraphNode<Self> {
        return TransposeGraphNode(base: self)
    }
}

/// Graph representation storing all forward and backward edges
public protocol BidirectionalEdgeSet {
    associatedtype Node : AnyObject
    func predecessors(of node: Node) -> ObjectSet<Node>
    func successors(of node: Node) -> ObjectSet<Node>
}

// MARK: - Transpose
extension BidirectionalEdgeSet {
    public var transpose: TransposeEdgeSet<Self> {
        return TransposeEdgeSet(base: self)
    }
}

/// Transpose edge set
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

/// Directed graph
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
    /// Insert node to the graph
    mutating func insertNode(_ node: Node) {
        if contains(node) { return }
        entries[node] = Entry()
    }

    /// Insert edge to the graph
    mutating func insertEdge(from src: Node, to dest: Node) {
        insertNode(src)
        entries[src]!.successors.insert(dest)
        insertNode(dest)
        entries[dest]!.predecessors.insert(src)
    }

    /// Remove everything from the graph
    mutating func removeAll() {
        entries.removeAll()
    }
}

// MARK: - Query
public extension DirectedGraph {

    /// Predecessors of the node
    func predecessors(of node: Node) -> ObjectSet<Node> {
        return self[node].predecessors
    }

    /// Successors of the node
    func successors(of node: Node) -> ObjectSet<Node> {
        return self[node].successors
    }

    /// Returns the graph node entry for the node
    subscript(node: Node) -> Entry {
        guard let entry = entries[node] else {
            preconditionFailure("Node not in the graph")
        }
        return entry
    }

    /// Does the graph contain this node?
    func contains(_ node: Node) -> Bool {
        return entries.keys.contains(node)
    }

    /// Does the graph contain this edge?
    func containsEdge(from src: Node, to dest: Node) -> Bool {
        return successors(of: src).contains(dest)
    }

    /// Does this node immediately precede the other?
    func immediatelyPrecedes(_ firstNode: Node, _ secondNode: Node) -> Bool {
        return predecessors(of: secondNode).contains(firstNode)
    }

    /// Does this node immediately succeed the other?
    func immediatelySucceeds(_ firstNode: Node, _ secondNode: Node) -> Bool {
        return successors(of: secondNode).contains(secondNode)
    }

    /// Does this node succeed the other?
    func precedes(_ firstNode: Node, _ secondNode: Node) -> Bool {
        guard contains(firstNode) else { return false }
        let secondPreds = predecessors(of: secondNode)
        return secondPreds.contains(firstNode)
            || secondPreds.contains(where: { precedes($0, secondNode) })
    }

    /// Does this node succeed the other?
    func succeeds(_ firstNode: Node, _ secondNode: Node) -> Bool {
        guard contains(secondNode) else { return false }
        let secondSuccs = successors(of: secondNode)
        return secondSuccs.contains(firstNode)
            || secondSuccs.contains(where: { succeeds(firstNode, $0) })
    }

    /// Is this node a source?
    func isSource(_ node: Node) -> Bool {
        return predecessors(of: node).isEmpty
    }

    /// Is this node a leaf?
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

    /// Recursively insert all vertices and edges to the graph by traversing
    /// forward from a source vertex
    ///
    /// - Parameter node: source vertex
    mutating func insertAll(fromSource node: Node) {
        var visited: ObjectSet<Node> = []
        func insertAll(fromSource node: Node) {
            for succ in node.successors where !visited.contains(succ) {
                visited.insert(succ)
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

    /// Recursively insert all vertices and edges to the graph by traversing 
    /// backward from a leaf vertex
    ///
    /// - Parameter node: leaf vertex
    mutating func insertAll(fromLeaf node: Node) {
        var visited: ObjectSet<Node> = []
        func insertAll(fromLeaf node: Node) {
            for pred in node.predecessors {
                visited.insert(pred)
                insertEdge(from: node, to: node)
                insertAll(fromLeaf: pred)
            }
        }
        insertAll(fromLeaf: node)
    }

}
