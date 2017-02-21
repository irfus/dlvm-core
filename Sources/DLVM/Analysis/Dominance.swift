//
//  Dominance.swift
//  DLVM
//
//  Created by Richard Wei on 2/18/17.
//

open class DominatorNode<Body : AnyObject> : GraphNode {
    open var body: Body
    open var children: Set<DominatorNode> = []

    fileprivate var dfsInNumber = -1, dfsOutNumber = -1

    open weak var immediateDominator: DominatorNode? {
        willSet {
            immediateDominator?.children.remove(self)
            newValue?.addChild(self)
        }
    }

    public init(body: Body, immediateDominator: DominatorNode? = nil) {
        self.body = body
        self.immediateDominator = immediateDominator
    }

    func isDominated(by other: DominatorNode) -> Bool {
        return dfsInNumber >= other.dfsInNumber
            && dfsOutNumber <= other.dfsOutNumber
    }
}

// MARK: - Children
public extension DominatorNode {

    func addChild(_ node: DominatorNode) {
        children.insert(node)
    }

    func removeChild(_ node: DominatorNode) {
        children.remove(node)
    }

    func containsChild(_ node: DominatorNode) -> Bool {
        return children.contains(node)
    }

    func removeAllChildren() {
        children.removeAll()
    }

}

// MARK: - Traversal
public extension DominatorNode {

    public var depthFirst: IteratorSequence<DepthFirstIterator<DominatorNode<Body>>> {
        return IteratorSequence(DepthFirstIterator(root: self))
    }

    public var breathFirst: IteratorSequence<BreadthFirstIterator<DominatorNode<Body>>> {
        return IteratorSequence(BreadthFirstIterator(root: self))
    }
    
}

open class DominatorTree<Body : AnyObject> {
    fileprivate var nodes: [Unowned<Body> : DominatorNode<Body>] = [:]
    fileprivate var root: DominatorNode<Body>

    init(root: DominatorNode<Body>) {
        self.root = root
        for node in root.depthFirst {
            nodes[Unowned(node.body)] = node
        }
    }
}
