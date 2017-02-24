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

extension Tree {
    public var preorder: IteratorSequence<GraphIterator<Tree<Element>>> {
        return IteratorSequence(GraphIterator(root: self, order: .preorder))
    }

    public var postorder: IteratorSequence<GraphIterator<Tree<Element>>> {
        return IteratorSequence(GraphIterator(root: self, order: .postorder))
    }

    public var levelOrder: IteratorSequence<GraphIterator<Tree<Element>>> {
        return IteratorSequence(GraphIterator(root: self, order: .breadthFirst))
    }
}
