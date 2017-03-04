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

public final class Function : Named, IRCollection, IRSubUnit {

    public typealias Element = Section

    public var name: String
    public var result: Argument?
    public var arguments: OrderedMapSet<Def<Argument>> = []
    public var isDifferentiable: Bool
    public unowned var parent: Module
    
    public var elements: OrderedMapSet<Section> = []
    public internal(set) var analysisManager: AnalysisManager<Function> = AnalysisManager()
    public internal(set) var transformManager: TransformManager<Function> = TransformManager()
    
    public unowned var top: Section {
        if let entry = elements["top"] {
            return entry
        }
        let bb = Section(asTopOf: self)
        elements.insert(bb, at: 0)
        return bb
    }

    public unowned var entry: BasicBlock {
        return top.entry
    }

    public init(name: String, arguments: [(String, Argument)], result: Argument?,
                isDifferentiable: Bool, parent: Module) {
        self.name = name
        self.arguments.append(contentsOf: arguments.map(Def.init))
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

}

// MARK: - Control flow
extension Function {

    open var allInstructions: LazyCollection<FlattenCollection<FlattenBidirectionalCollection<Function>>> {
        return allBasicBlocks.joined()
    }

    open var allBasicBlocks: LazyCollection<FlattenBidirectionalCollection<Function>> {
        return lazy.joined()
    }

    open weak var endBlock: BasicBlock? {
        return top.endBlock
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
