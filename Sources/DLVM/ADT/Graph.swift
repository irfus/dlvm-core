//
//  Graph.swift
//  DLVM
//
//  Created by Richard Wei on 2/19/17.
//
//

public protocol GraphNode {
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

public extension BidirectionalGraphNode
    where SuccessorSequence.Iterator.Element == Self,
          PredecessorSequence.Iterator.Element == Self {
    func reversedGraphNode() -> ReversedGraphNode<Self> {
        return ReversedGraphNode(base: self)
    }
}

public class ReversedGraphNode<Base : BidirectionalGraphNode> : BidirectionalGraphNode
    where Base.SuccessorSequence.Iterator.Element == Base,
          Base.PredecessorSequence.Iterator.Element == Base
{
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

