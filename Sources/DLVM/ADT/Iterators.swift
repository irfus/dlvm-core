//
//  Iterators.swift
//  DLVM
//
//  Created by Richard Wei on 2/19/17.
//
//

import class Foundation.NSMutableSet

public enum TraversalOrder {
    case preorder, postorder, breadthFirst
}

public struct GraphNodeIterator<Node : ForwardGraphNode> : IteratorProtocol
    where Node.SuccessorCollection.Iterator.Element == Node {

    private var pre: [Node] = []
    private var post: [Node] = []
    private var visited = NSMutableSet()
    public let order: TraversalOrder

    public init(root: Node?, order: TraversalOrder) {
        if let root = root {
            pre.append(root)
        }
        self.order = order
    }

    public mutating func next() -> Node? {
        switch order {
        case .breadthFirst:
            if pre.isEmpty { return nil }
            let node = pre.removeFirst()
            for child in node.successors where !visited.contains(child) {
                pre.append(child)
            }
            visited.add(node)
            return node

        case .preorder:
            if pre.isEmpty { return nil }
            let node = pre.removeLast()
            for child in node.successors.reversed() where !visited.contains(child) {
                pre.append(child)
            }
            visited.add(node)
            return node

        case .postorder:
            if pre.isEmpty { return post.popLast() }
            let node = pre.removeLast()
            for child in node.successors where !visited.contains(child) {
                pre.append(child)
            }
            post.append(node)
            visited.add(node)
            return next()
        }
    }
}

public struct TransposeGraphNodeIterator<Node : BackwardGraphNode> : IteratorProtocol
    where Node.PredecessorCollection.Iterator.Element == Node {

    private var pre: [Node] = []
    private var post: [Node] = []
    private var visited = NSMutableSet()
    public let order: TraversalOrder

    public init(root: Node?, order: TraversalOrder) {
        if let root = root {
            pre.append(root)
        }
        self.order = order
    }

    public mutating func next() -> Node? {
        switch order {
        case .breadthFirst:
            if pre.isEmpty { return nil }
            let node = pre.removeFirst()
            for child in node.predecessors where !visited.contains(child) {
                pre.append(child)
            }
            visited.add(node)
            return node

        case .preorder:
            if pre.isEmpty { return nil }
            let node = pre.removeLast()
            for child in node.predecessors.reversed() where !visited.contains(child) {
                pre.append(child)
            }
            visited.add(node)
            return node

        case .postorder:
            if pre.isEmpty { return post.popLast() }
            let node = pre.removeLast()
            for child in node.predecessors where !visited.contains(child) {
                pre.append(child)
            }
            post.append(node)
            visited.add(node)
            return next()
        }
    }
    
}

public extension ForwardGraphNode where SuccessorCollection.Iterator.Element == Self {
    func traversed(in order: TraversalOrder) -> IteratorSequence<GraphNodeIterator<Self>> {
        return IteratorSequence(GraphNodeIterator(root: self, order: order))
    }

    var preorder: IteratorSequence<GraphNodeIterator<Self>> {
        return traversed(in: .preorder)
    }

    var postorder: IteratorSequence<GraphNodeIterator<Self>> {
        return traversed(in: .postorder)
    }

    var breadthFirst: IteratorSequence<GraphNodeIterator<Self>> {
        return traversed(in: .breadthFirst)
    }
}

public extension BackwardGraphNode
    where PredecessorCollection.Iterator.Element == Self {
    func transposeTraversed(in order: TraversalOrder) -> IteratorSequence<TransposeGraphNodeIterator<Self>> {
        return IteratorSequence(TransposeGraphNodeIterator(root: self, order: order))
    }
}

public struct DirectedGraphIterator<Base : BidirectionalEdgeSet> : IteratorProtocol {
    public typealias Element = Base.Node

    private var pre: [Base.Node] = []
    private var post: [Base.Node] = []
    private var visited: ObjectSet<Base.Node> = []
    public let order: TraversalOrder
    public let base: Base

    public init(base: Base, source: Base.Node?, order: TraversalOrder) {
        self.base = base
        if let source = source {
            pre.append(source)
        }
        self.order = order
    }
    
    public mutating func next() -> Base.Node? {
        switch order {
        case .breadthFirst:
            if pre.isEmpty { return nil }
            let node = pre.removeFirst()
            for child in base.successors(of: node) where !visited.contains(child) {
                pre.append(child)
            }
            visited.insert(node)
            return node
            
        case .preorder:
            if pre.isEmpty { return nil }
            let node = pre.removeLast()
            for child in base.successors(of: node).reversed() where !visited.contains(child) {
                pre.append(child)
            }
            visited.insert(node)
            return node
            
        case .postorder:
            if pre.isEmpty { return post.popLast() }
            let node = pre.removeLast()
            for child in base.successors(of: node) where !visited.contains(child) {
                pre.append(child)
            }
            post.append(node)
            visited.insert(node)
            return next()
        }
        
    }
}

// MARK: - Iterators
public extension BidirectionalEdgeSet {
    public func traversed(from source: Node, in order: TraversalOrder) -> IteratorSequence<DirectedGraphIterator<Self>> {
        return IteratorSequence(DirectedGraphIterator(base: self, source: source, order: order))
    }
}
