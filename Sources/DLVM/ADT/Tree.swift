//
//  Tree.swift
//  DLVM
//
//  Created by Richard Wei on 2/19/17.
//
//

public struct Tree<Element> : GraphNode {
    public var value: Element
    public var children: [Tree] = []

    public init(value: Element) {
        self.value = value
    }
}

extension Tree : Sequence {
    public typealias Iterator = DepthFirstIterator<Tree<Element>>

    public func makeIterator() -> Iterator {
        return Iterator(root: self)
    }
}
