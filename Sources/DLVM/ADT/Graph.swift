//
//  Graph.swift
//  DLVM
//
//  Created by Richard Wei on 2/19/17.
//
//

public protocol OutgoingGraphNode {
    associatedtype SuccessorCollection: Collection // where SuccessorCollection.Iterator.Element == Self
    var successors: SuccessorCollection { get }
}

public protocol IncomingGraphNode {
    associatedtype PredecessorCollection: Collection
    var predecessors: PredecessorCollection { get }
}

public extension OutgoingGraphNode {
    var isLeaf: Bool {
        return successors.isEmpty
    }
}

public extension IncomingGraphNode {
    var isSource: Bool {
        return predecessors.isEmpty
    }
}

public typealias BidirectionalGraphNode = OutgoingGraphNode & IncomingGraphNode

public extension OutgoingGraphNode
    where Self : IncomingGraphNode,
          SuccessorCollection.Iterator.Element == Self,
          Self.PredecessorCollection.Iterator.Element == Self {
    func reversedGraphNode() -> ReversedGraphNode<Self> {
        return ReversedGraphNode(base: self)
    }
}

public class ReversedGraphNode<Base : BidirectionalGraphNode> : BidirectionalGraphNode
    where Base.SuccessorCollection.Iterator.Element == Base,
          Base.PredecessorCollection.Iterator.Element == Base
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

