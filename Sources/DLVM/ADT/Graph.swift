//
//  Graph.swift
//  DLVM
//
//  Created by Richard Wei on 2/19/17.
//
//

public protocol GraphNode {
    var children: [Self] { get }
}

public extension GraphNode {
    var isLeaf: Bool {
        return children.isEmpty
    }
}
