//
//  Intrinsics.swift
//  DLVM
//
//  Created by Richard Wei on 2/13/17.
//
//

public final class Function : Named, IRCollection, IRSubUnit {

    public typealias Element = BasicBlock

    public var name: String
    public var result: Type
    public var arguments: OrderedMapSet<Argument> = []
    public var isDifferentiable: Bool
    public unowned var parent: Module
    
    public var elements: OrderedMapSet<BasicBlock> = []
    public internal(set) var analysisManager: AnalysisManager<Function> = AnalysisManager()
    public internal(set) var transformManager: TransformManager<Function> = TransformManager()

    public unowned var entry: BasicBlock {
        if let entry = elements["entry"] {
            return entry
        }
        let bb = BasicBlock(asEntryOf: self)
        elements.insert(bb, at: 0)
        return bb
    }

    public init(name: String, arguments: [(String?, Type)],
                result: Type, isDifferentiable: Bool, parent: Module) {
        self.name = name
        self.arguments.append(contentsOf: arguments.map(Argument.init))
        self.result = result
        self.isDifferentiable = isDifferentiable
        self.parent = parent
        _ = entry
    }

}

// MARK: - Arguments
public extension Function {

    func acceptsArguments<C : Collection>(_ types: C) -> Bool where C.Iterator.Element == Type {
        return types.elementsEqual(arguments.map{$0.type})
    }

    func argument(named name: String) -> Argument? {
        return arguments.element(named: name)
    }

    func containsArgument(named name: String) -> Bool {
        return arguments.containsElement(named: name)
    }
    
}

// MARK: - Control flow
extension Function {

    open var instructions: LazyCollection<FlattenBidirectionalCollection<Function>> {
        return lazy.joined()
    }

    open func instruction(named name: String) -> Instruction? {
        for bb in self {
            if let inst = bb.element(named: name) {
                return inst
            }
        }
        return nil
    }

    open func containsInstruction(named name: String) -> Bool {
        return instruction(named: name) != nil
    }

    open func containsName(_ name: String) -> Bool {
        return containsElement(named: name) || contains(where: {
            $0.containsArgument(named: name) || $0.containsElement(named: name)
        })
    }

}
