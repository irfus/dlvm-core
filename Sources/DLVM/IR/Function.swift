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

open class Function : Named, IRCollection, IRObject {
    public typealias Element = BasicBlock

    public var name: String
    public var arguments: [Def<Argument>]
    public var result: Argument?
    public var forwardPass = OrderedKVSet<BasicBlock>()
    public var backwardPass: OrderedKVSet<BasicBlock>?

    public lazy var parametricBackwardPasses: [OrderedKVSet<BasicBlock>] =
        self.arguments.map { _ in [] }

    public weak var parent: Module?

    public init(name: String, arguments: [(String, Argument)], result: Argument?) {
        self.name = name
        self.arguments = arguments.map { name, arg in
            Def<Argument>(name: name, value: arg)
        }
        self.result = result
    }

}

// MARK: - Forward basic block management
extension Function {

    open func append(_ basicBlock: BasicBlock) {
        if let existingBlock = self.basicBlock(named: basicBlock.name) {
            remove(existingBlock)
        }
        forwardPass.append(basicBlock)
        basicBlock.parent = self
    }

    open func insert(_ basicBlock: BasicBlock, after previous: BasicBlock) {
        if let existingBlock = self.basicBlock(named: basicBlock.name) {
            remove(existingBlock)
        }
        forwardPass.insert(basicBlock, after: previous)
        basicBlock.parent = self
    }

    open func index(of basicBlock: BasicBlock) -> Int? {
        return forwardPass.index(of: basicBlock)
    }

    open func remove(_ basicBlock: BasicBlock) {
        forwardPass.remove(basicBlock)
        basicBlock.parent = nil
    }

    open func basicBlock(named name: String) -> BasicBlock? {
        return forwardPass.element(named: name)
    }

    open func containsBasicBlock(named name: String) -> Bool {
        return forwardPass.containsValue(named: name)
    }

    open func contains(_ basicBlock: BasicBlock) -> Bool {
        return forwardPass.contains(basicBlock)
    }

    open var elements: [BasicBlock] {
        return Array(forwardPass)
    }

    open func localValue(named name: String) -> Use? {
        for bb in forwardPass {
            if let oper = bb.operation(named: name) {
                return Use(kind: .local(oper))
            }
        }
        return nil
    }
    
}

// MARK: - Control Flow Graph
extension Function {

    open weak var entry: BasicBlock? {
        return forwardPass.element(named: "entry")
    }

    open var instructions: [Instruction] {
        return forwardPass.lazy.flatMap{$0.instructions}
    }

    open var depthFirst: IteratorSequence<DepthFirstIterator<BasicBlock>> {
        return IteratorSequence(DepthFirstIterator(root: entry))
    }

    open var breathFirst: IteratorSequence<BreathFirstIterator<BasicBlock>> {
        return IteratorSequence(BreathFirstIterator(root: entry))
    }

}
