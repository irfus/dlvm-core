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

public enum DifferentiationVariable {
    case argument(Def<Argument>)
    case global(Def<Placeholder>)

    public var definition: AnyDef {
        switch self {
            case .global(let def): return def
            case .argument(let def): return def
        }
    }
}

extension DifferentiationVariable : Hashable {
    public static func ==(lhs: DifferentiationVariable, rhs: DifferentiationVariable) -> Bool {
        switch (lhs, rhs) {
        case let (.argument(x), .argument(y)): return x === y
        case let (.global(x), .global(y)): return x === y
        default: return false
        }
    }
    
    public var hashValue: Int {
        switch self {
        case .argument(let def): return ObjectIdentifier(def).hashValue
        case .global(let def): return ObjectIdentifier(def).hashValue
        }
    }
}

open class Function : Named, IRCollection, IRObject {
    public typealias Element = BasicBlock

    public var name: String
    public var arguments: OrderedKVSet<Def<Argument>>
    public var result: Argument?
    public var forwardPass = OrderedKVSet<BasicBlock>()
    public internal(set) weak var returnBlock: BasicBlock?

    public var backwardPasses: [DifferentiationVariable : OrderedKVSet<BasicBlock>] = [:]

    public weak var parent: Module?

    public init(name: String, arguments: [(String, Argument)], result: Argument?) {
        self.name = name
        self.arguments = []
        for (name, arg) in arguments {
            let def = Def<Argument>(name: name, value: arg)
            self.arguments.append(def)
        }
        self.result = result
    }

}

// MARK: - Argument accessors
extension Function {

    open func argument(named name: String) -> Def<Argument>? {
        return arguments.element(named: name)
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
        /// If it contains an exit instruction, remember this block
        /// as the exit
        if basicBlock.isReturn {
            returnBlock = basicBlock
        }
    }

    open func insert(_ basicBlock: BasicBlock, after previous: BasicBlock) {
        if let existingBlock = self.basicBlock(named: basicBlock.name) {
            remove(existingBlock)
        }
        forwardPass.insert(basicBlock, after: previous)
        basicBlock.parent = self
        /// If it contains an exit instruction, remember this block
        /// as the exit
        if basicBlock.isReturn {
            returnBlock = basicBlock
        }
    }

    open func index(of basicBlock: BasicBlock) -> Int? {
        return forwardPass.index(of: basicBlock)
    }

    open func remove(_ basicBlock: BasicBlock) {
        forwardPass.remove(basicBlock)
        basicBlock.parent = nil
        /// If it's the currently remembered exit, forget it
        if returnBlock === basicBlock {
            returnBlock = nil
        }
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

}

// MARK: - Iterator of all forward and backward passes
extension Function {

    open var allBasicBlocks: FlattenBidirectionalCollection<[OrderedKVSet<BasicBlock>]> {
        return ([forwardPass] + backwardPasses.values).joined()
    }

}
