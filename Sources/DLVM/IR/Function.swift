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
    public var arguments: [Def<Argument>]
    public var result: Argument?
    public var forwardSection = OrderedNamedObjectSet<BasicBlock>()
    public var backwardSection = OrderedNamedObjectSet<BasicBlock>()

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
        forwardSection.insert(basicBlock)
        basicBlock.parent = self
    }

    open func insert(_ basicBlock: BasicBlock, after previous: BasicBlock) {
        if let existingBlock = self.basicBlock(named: basicBlock.name) {
            remove(existingBlock)
        }
        forwardSection.insert(basicBlock, after: previous)
        basicBlock.parent = self
    }

    open func index(of basicBlock: BasicBlock) -> Int? {
        return forwardSection.index(of: basicBlock)
    }

    open func remove(_ basicBlock: BasicBlock) {
        forwardSection.remove(basicBlock)
        basicBlock.parent = nil
    }

    open func basicBlock(named name: String) -> BasicBlock? {
        return forwardSection.element(named: name)
    }

    open func containsBasicBlock(named name: String) -> Bool {
        return forwardSection.containsValue(named: name)
    }

    open func contains(_ basicBlock: BasicBlock) -> Bool {
        return forwardSection.contains(basicBlock)
    }

    open var elements: [BasicBlock] {
        return Array(forwardSection)
    }
    
}

extension Function {

    open weak var forwardEntry: BasicBlock? {
        return forwardSection.element(named: "forward")
    }

    open weak var backwardEntry: BasicBlock? {
        return forwardSection.element(named: "backward")
    }

    open var instructions: [Instruction] {
        return forwardSection.lazy.flatMap{$0.instructions}
    }

    open func localValue(named name: String) -> Use? {
        for bb in forwardSection {
            if let oper = bb.operation(named: name) {
                return Use(kind: .local(oper))
            }
        }
        return nil
    }

}
