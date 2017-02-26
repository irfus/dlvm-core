//
//  Graph.swift
//  DLVM
//
//  Created by Richard Wei on 2/19/17.
//
//

public protocol GraphNode : AnyObject {
    associatedtype SuccessorSequence: Sequence // where SuccessorSequence.Iterator.Element == Self
    var successors: SuccessorSequence { get }
}

public protocol BidirectionalGraphNode : GraphNode {
    associatedtype PredecessorSequence: Sequence // where PredecessorSequence.Iterator.Element == Self
    var predecessors: PredecessorSequence { get }
}

public extension GraphNode where SuccessorSequence: Collection {
    var isLeaf: Bool {
        return successors.isEmpty
    }
}

public extension GraphNode where SuccessorSequence: Sequence, SuccessorSequence.Iterator.Element == Self {
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
