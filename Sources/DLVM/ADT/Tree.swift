//
//  Tree.swift
//  DLVM
//
//  Created by Richard Wei on 2/19/17.
//
//

public struct Tree<Element> : GraphNode {
    public var value: Element
    public var children: [Tree]

    public init(value: Element, children: [Tree] = []) {
        self.value = value
        self.children = children
    }
}

extension Tree : Sequence {
    public typealias Iterator = DepthFirstIterator<Tree<Element>>

    public func makeIterator() -> Iterator {
        return Iterator(root: self)
    }

    public var preorder: IteratorSequence<DepthFirstIterator<Tree<Element>>> {
        return IteratorSequence(DepthFirstIterator(root: self))
    }

    public var levelOrder: IteratorSequence<BreadthFirstIterator<Tree<Element>>> {
        return IteratorSequence(BreadthFirstIterator(root: self))
    }
}
