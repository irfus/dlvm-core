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
        public unowned var parent: Function

        public init(variable: DifferentiationVariable? = nil, parent: Function) {
            self.variable = variable
            self.parent = parent
        }

    }

    public typealias Element = BasicBlock

    public var name: String
    public var arguments: OrderedMapSet<Def<Argument>>
    public var result: Argument?
    public lazy var forwardPass: Section? = Section(parent: self)
    public var backwardPasses: [DifferentiationVariable : Section] = [:]
    public unowned var parent: Module

    public var elements: OrderedMapSet<BasicBlock> {
        get { return forwardPass?.elements ?? [] }
        set { forwardPass?.elements = newValue }
    }

    public init(name: String, arguments: [(String, Argument)], result: Argument?, parent: Module) {
        self.name = name
        self.arguments = []
        for (name, arg) in arguments {
            let def = Def<Argument>(name: name, value: arg)
            self.arguments.append(def)
        }
        self.result = result
        self.parent = parent
    }

}

// MARK: - Accessors
extension Function {

    open func argument(named name: String) -> Def<Argument>? {
        return arguments.element(named: name)
    }

    /// Returns forward pass if there's an existing one, o
    /// otherwise create a new one
    ///
    /// - Returns: forward pass
    open func makeForwardPass() -> Section {
        return forwardPass ?? {
            let forward = Section(parent: self)
            forwardPass = forward
            return forward
        }()
    }

    open func backwardPass(withRespectTo variable: DifferentiationVariable) -> Section? {
        return backwardPasses[variable]
    }

}

// MARK: - Forward basic block management
extension Function {

    open func localValue(named name: String) -> Use? {
        guard let forwardPass = forwardPass else { return nil }
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

    open weak var endBlock: BasicBlock? {
        return first(where: { block in
            block.isReturn || block.isYielding
        })
    }

    open var isForward: Bool {
        return parent.forwardPass === self
    }

    open var isBackward: Bool {
        return parent.backwardPasses.values.contains(self)
    }

    open var differentiationVariables: [DifferentiationVariable] {
        var variables: [DifferentiationVariable] = []
        for bb in self {
            variables.append(contentsOf: bb.usedPlaceholders.map{.global($0)})
            variables.append(contentsOf: bb.usedArguments.map{.argument($0)})
        }
        return variables
    }

}

// MARK: - Control flow
extension Function {

    open weak var entry: BasicBlock? {
        return forwardPass?.entry
    }

    open var allInstructions: [Instruction] {
        return allBasicBlocks.flatMap { $0 }
    }

    open var allBasicBlocks: FlattenBidirectionalCollection<[Section]> {
        return ([forwardPass].flatMap{$0} + backwardPasses.values).joined()
    }

    open weak var endBlock: BasicBlock? {
        return forwardPass?.endBlock
    }

    open func instruction(named name: String) -> Instruction? {
        for bb in allBasicBlocks {
            if let inst = bb.element(named: name) {
                return inst
            }
        }
        return nil
    }

    open func containsInstruction(named name: String) -> Bool {
        return instruction(named: name) != nil
    }

}
