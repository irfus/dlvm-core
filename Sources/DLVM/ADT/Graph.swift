//
//  Graph.swift
//  DLVM
//
//  Created by Richard Wei on 2/19/17.
//
//

public protocol GraphNode {
    associatedtype Children : Sequence
    var children: Children { get }
}

public extension GraphNode where Children : Collection {
    var isLeaf: Bool {
        return children.isEmpty
    }
}

public extension  GraphNode where Children : Collection, Children.Iterator.Element == Self {
    var preorder: IteratorSequence<GraphIterator<Self>> {
        return IteratorSequence(GraphIterator(root: self, order: .preorder))
    }

    var postorder: IteratorSequence<GraphIterator<Self>> {
        return IteratorSequence(GraphIterator(root: self, order: .postorder))
    }

    var breadthFirst: IteratorSequence<GraphIterator<Self>> {
        return IteratorSequence(GraphIterator(root: self, order: .breadthFirst))
    }
}
