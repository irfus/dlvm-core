//
//  Iterators.swift
//  DLVM
//
//  Created by Richard Wei on 2/19/17.
//
//

public struct GraphIterator<Node : GraphNode> : IteratorProtocol
    where Node.SuccessorSequence.Iterator.Element == Node {

    public enum Order {
        case preorder, postorder, breadthFirst
    }

    private var pre: [Node] = []
    private var post: [Node] = []
    private var visited: Set<Node> = []
    public let order: Order

    public init(root: Node?, order: Order) {
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
            visited.insert(node)
            return node

        case .preorder:
            if pre.isEmpty { return nil }
            let node = pre.removeLast()
            for child in node.successors.reversed() where !visited.contains(child) {
                pre.append(child)
            }
            visited.insert(node)
            return node

        case .postorder:
            if pre.isEmpty { return post.popLast() }
            let node = pre.removeLast()
            for child in node.successors where !visited.contains(child) {
                pre.append(child)
            }
            post.append(node)
            visited.insert(node)
            return next()
        }
    }
}

