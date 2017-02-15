//
//  Intrinsics.swift
//  DLVM
//
//  Created by Richard Wei on 2/13/17.
//
//

public struct Argument : Value {
    public var shape: TensorShape
    public var type: DataType
    public var broadcasting: Bool
    public static var scope: Scope = .local
}

open class Function : Named, IRCollection {
    public typealias Element = BasicBlock

    public var name: String
    public var argument: Def<Argument>?
    public var result: Argument?
    public var basicBlocks = OrderedNamedObjectSet<BasicBlock>()

    public weak var parent: Module?

    public init(name: String, argument: Def<Argument>?, result: Argument?) {
        self.name = name
        self.argument = argument
        self.result = result
    }
    
}


// MARK: - Basic blocks
extension Function {

    open func append(_ basicBlock: BasicBlock) {
        if let existingBlock = self.basicBlock(named: basicBlock.name) {
            remove(existingBlock)
        }
        basicBlocks.insert(basicBlock)
        basicBlock.parent = self
    }

    open func index(of basicBlock: BasicBlock) -> Int? {
        return basicBlocks.index(of: basicBlock)
    }

    open func remove(_ basicBlock: BasicBlock) {
        basicBlocks.remove(basicBlock)
        basicBlock.parent = nil
    }

    open func basicBlock(named name: String) -> BasicBlock? {
        return basicBlocks.element(named: name)
    }

    open func containsBasicBlock(named name: String) -> Bool {
        return basicBlocks.containsValue(named: name)
    }

    open func contains(_ basicBlock: BasicBlock) -> Bool {
        return basicBlocks.contains(basicBlock)
    }

    open var elements: [BasicBlock] {
        return Array(basicBlocks)
    }
    
}

extension Function {

    open weak var forwardEntry: BasicBlock? {
        return basicBlocks.element(named: "forward")
    }

    open weak var backwardEntry: BasicBlock? {
        return basicBlocks.element(named: "backward")
    }

    open var instructions: [Instruction] {
        return basicBlocks.lazy.flatMap{$0.instructions}
    }

    /// TODO: Make substitutions of basic block references in branch instructions
//
//    func substitution(_ actualArgument: Use) -> [BasicBlock] {
//        let target = Use(kind: .argument(argument))
//        /// Copy BBs
//        let newBBs = basicBlocks.map { BasicBlock(name: $0.name) }
//        for (bb, newBB) in zip(basicBlocks, newBBs) {
//            for inst in bb {
//
//            }
//            bb.instructions.map { $0.substituting(actualArgument, for: target) }
//        }
//        return newBBs
//    }

}
