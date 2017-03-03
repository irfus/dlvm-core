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

public final class Section : IRCollection, IRSubUnit, Named, BackwardGraphNode {
    
    public typealias Element = BasicBlock
    public typealias PredecessorSequence = ObjectSet<Section>

    public var name: String
    public var destination: Def<Argument>?
    public var predecessors: ObjectSet<Section>
    public var elements: OrderedMapSet<BasicBlock> = []
    public unowned var parent: Function
    public internal(set) var analysisManager: AnalysisManager<Section> = AnalysisManager()
    
    public init(name: String,
                dependencies: ObjectSet<Section>,
                destination: Def<Argument>? = nil,
                parent: Function) {
        self.name = name
        self.predecessors = dependencies
        self.destination = destination
        self.parent = parent
    }
    
}

public final class Function : Named, IRCollection, IRSubUnit {

    public typealias Element = Section

    public var name: String
    public var arguments: OrderedMapSet<Def<Argument>>
    public var result: Argument?
    public var isDifferentiable: Bool
    public var elements: OrderedMapSet<Section> = []
    public unowned var parent: Module
    public var analysisManager: AnalysisManager<Function> = AnalysisManager()
    
    public weak var top: Section? {
        return elements.element(named: "top")
    }

    public init(name: String, arguments: [(String, Argument)], result: Argument?, isDifferentiable: Bool, parent: Module) {
        self.name = name
        self.arguments = []
        for (name, arg) in arguments {
            let def = Def<Argument>(name: name, value: arg)
            self.arguments.append(def)
        }
        self.result = result
        self.isDifferentiable = isDifferentiable
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
    open func makeForwardPass(dependingOn dependencies: ObjectSet<Section>) -> Section {
        return top ?? {
            let forward = Section(name: "forward", dependencies: dependencies, parent: self)
            append(forward)
            return forward
        }()
    }

}

// MARK: - Forward basic block management
extension Function {

    open func localValue(named name: String) -> Use? {
        guard let forwardPass = top else { return nil }
        for bb in forwardPass {
            if case let .operation(oper)? = bb.element(named: name)?.kind {
                return Use(kind: .local(oper))
            }
        }
        return nil
    }

}

// MARK: - Control flow
extension Section {

    open weak var entry: BasicBlock? {
        return elements.element(named: "entry")
    }

    open weak var endBlock: BasicBlock? {
        return first(where: { block in
            block.isReturn || block.isYielding
        })
    }

    open var module: Module {
        return parent.parent
    }

    open var isForward: Bool {
        return parent.top === self
    }

}

// MARK: - Control flow
extension Function {

    open var allInstructions: [Instruction] {
        return allBasicBlocks.flatMap { $0 }
    }

    open var allBasicBlocks: FlattenBidirectionalCollection<Function> {
        return joined()
    }

    open weak var endBlock: BasicBlock? {
        return top?.endBlock
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
