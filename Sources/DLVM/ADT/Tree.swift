//
//  Tree.swift
//  DLVM
//
//  Created by Richard Wei on 2/19/17.
//
//

public class TreeNode<Element> : ForwardGraphNode {
    public var value: Element
    public var successors: [TreeNode]

    public init(value: Element, successors: [TreeNode] = []) {
        self.value = value
        self.successors = successors
    }
}
