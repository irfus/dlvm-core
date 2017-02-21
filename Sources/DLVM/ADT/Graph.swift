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
