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

public struct GraphIterator<Node : GraphNode> : IteratorProtocol
    where Node.SuccessorSequence.Iterator.Element == Node {

    private var pre: [Node] = []
    private var post: [Node] = []
    private var visited: NSMutableSet = []
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

public struct TransposeGraphIterator<Node : BidirectionalGraphNode> : IteratorProtocol
    where Node.SuccessorSequence.Iterator.Element == Node,
          Node.PredecessorSequence.Iterator.Element == Node {

    private var pre: [Node] = []
    private var post: [Node] = []
    private var visited: NSMutableSet = []
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

public extension GraphNode where SuccessorSequence.Iterator.Element == Self {
    func traversed(in order: TraversalOrder) -> IteratorSequence<GraphIterator<Self>> {
        return IteratorSequence(GraphIterator(root: self, order: order))
    }

    var preorder: IteratorSequence<GraphIterator<Self>> {
        return traversed(in: .preorder)
    }

    var postorder: IteratorSequence<GraphIterator<Self>> {
        return traversed(in: .postorder)
    }

    var breadthFirst: IteratorSequence<GraphIterator<Self>> {
        return traversed(in: .breadthFirst)
    }
}

public extension BidirectionalGraphNode where SuccessorSequence.Iterator.Element == Self,
                                              PredecessorSequence.Iterator.Element == Self {
    func transposeTraversed(in order: TraversalOrder) -> IteratorSequence<TransposeGraphIterator<Self>> {
        return IteratorSequence(TransposeGraphIterator(root: self, order: order))
    }
}
