//
// Created by Richard Wei on 2/18/17.
//

public struct DominatorNode {
    internal var reversePostorderID: Int
    internal var immediateDominator: Instruction
}

public struct DominatorTree {
    public var nodes: [BasicBlock]
}

public extension DominatorTree {

    public init?(function: Function) {
        guard let entry = function.entry else { return nil }
        /// TODO: implement
        fatalError()
    }

}
