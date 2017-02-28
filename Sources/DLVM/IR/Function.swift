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

open class Function : Named, IRCollection, IRUnit {
    
    open class Section : IRCollection, IRUnit {

        public typealias Element = BasicBlock

        public var variable: DifferentiationVariable?
        public var elements: OrderedMapSet<BasicBlock> = []
        public weak var parent: Function?
        
        public init(variable: DifferentiationVariable? = nil) {
            self.variable = variable
        }
        
    }

    public typealias Element = BasicBlock

    public var name: String
    public var arguments: OrderedMapSet<Def<Argument>>
    public var result: Argument?
    public let forwardPass = Section()
    public var backwardPasses: [DifferentiationVariable : Section] = [:]

    public var elements: OrderedMapSet<BasicBlock> {
        get {
            return forwardPass.elements
        }
        set {
            return forwardPass.elements = newValue
        }
    }

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

// MARK: - Evaluation pass access helpers
public extension OrderedMapSet where Element == BasicBlock {
    var entry: Element? {
        return element(named: "entry")
    }
}

// MARK: - Argument accessors
extension Function {

    open func argument(named name: String) -> Def<Argument>? {
        return arguments.element(named: name)
    }

    open func backwardPass(withRespectTo variable: DifferentiationVariable) -> Section? {
        return backwardPasses[variable]
    }
    
}

// MARK: - Forward basic block management
extension Function {

    open func localValue(named name: String) -> Use? {
        for bb in forwardPass {
            if case let .operation(oper)? = bb.element(named: name)?.kind {
                return Use(kind: .local(oper))
            }
        }
        return nil
    }
    
}

// MARK: - Control flow
extension Function.Section {

    open weak var entry: BasicBlock? {
        return elements.element(named: "entry")
    }

    open weak var returnBlock: BasicBlock? {
        return first(where: {$0.terminator?.isReturn ?? false})
    }
    
}

// MARK: - Control flow
extension Function {

    open weak var entry: BasicBlock? {
        return forwardPass.entry
    }

    open var allInstructions: [Instruction] {
        return allBasicBlocks.flatMap { $0 }
    }

    open var allBasicBlocks: FlattenBidirectionalCollection<[Section]> {
        return ([forwardPass] + backwardPasses.values).joined()
    }

    open weak var returnBlock: BasicBlock? {
        return forwardPass.returnBlock
    }

}
