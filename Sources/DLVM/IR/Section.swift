//
//  Section.swift
//  DLVM
//
//  Created by Richard Wei on 3/4/17.
//
//

public final class Section : IRCollection, IRSubUnit, Named, BackwardGraphNode {
    
    public typealias Element = BasicBlock
    public typealias PredecessorSequence = ObjectSet<Section>

    public var name: String
    public var derivation: Def<Argument>?
    public var predecessors: ObjectSet<Section>
    public var elements: OrderedMapSet<BasicBlock> = []
    public unowned var parent: Function
    public internal(set) var analysisManager: AnalysisManager<Section> = AnalysisManager()
    public internal(set) var transformManager: TransformManager<Section> = TransformManager()

    public unowned var entry: BasicBlock {
        if let entry = elements["entry"] {
            return entry
        }
        let bb = BasicBlock(asEntryOf: self)
        elements.insert(bb, at: 0)
        return bb
    }
    
    public init(name: String,
                dependencies: ObjectSet<Section>,
                derivation: Def<Argument>? = nil,
                parent: Function) {
        self.name = name
        self.predecessors = dependencies
        self.derivation = derivation
        self.parent = parent
    }

    internal convenience init(asTopOf parent: Function) {
        self.init(name: "top", dependencies: [], parent: parent)
    }

    internal convenience init(asDerivativeOf parent: Function) {
        self.init(name: "derivative", dependencies: [], parent: parent)
    }
    
}

// MARK: - Control flow
extension Section {

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
