//
//  Iterators.swift
//  DLVM
//
//  Created by Richard Wei on 2/19/17.
//
//

public protocol WhateverFirstIterator : IteratorProtocol {
    associatedtype Node : GraphNode
    init(root: Node?)
}

public struct DepthFirstIterator<Node : GraphNode> : WhateverFirstIterator {
    private var stack: [Node] = []
    private var visited: KVSet<Node> = []

    public init(root: Node?) {
        if let root = root {
            stack.append(root)
        }
    }
    
    public mutating func next() -> Node? {
        if stack.isEmpty { return nil }
        let node = stack.removeLast()
        for child in node.children.reversed() where !visited.contains(child) {
            stack.append(child)
        }
        visited.insert(node)
        return node
    }
}

public struct BreathFirstIterator<Node : GraphNode> : WhateverFirstIterator {
    private var stack: [Node] = []
    private var visited: KVSet<Node> = []
    
    public init(root: Node?) {
        if let root = root {
            stack.append(root)
        }
    }

    public mutating func next() -> Node? {
        if stack.isEmpty { return nil }
        let node = stack.removeFirst()
        visited.insert(node)
        for child in node.children where !visited.contains(child) {
            stack.append(child)
        }
        return node
    }
}
