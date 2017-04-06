//
//  BasicBlock.swift
//  DLVM
//
//  Copyright 2016-2017 Richard Wei.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

public class Argument : Value, Named, Definition, HashableByReference {
    public var name: String
    public var type: Type
    public weak var parent: BasicBlock?

    public init(name: String, type: Type) {
        self.name = name
        self.type = type
    }

    public func makeUse() -> Use {
        return .argument(type, self)
    }
}

public final class BasicBlock : IRCollection, IRSubUnit, Named {

    public typealias Element = Instruction

    /// Name of the basic block
    open var name: String
    open var arguments: OrderedMapSet<Argument> = []
    open var elements: OrderedMapSet<Instruction> = []
    open unowned var parent: Function
    public internal(set) var analysisManager: AnalysisManager<BasicBlock> = AnalysisManager()
    public internal(set) var transformManager: TransformManager<BasicBlock> = TransformManager()

    internal init<C: Collection>(name: String, arguments: C, parent: Function)
        where C.Iterator.Element == Argument
    {
        self.name = name
        self.arguments.append(contentsOf: arguments)
        self.parent = parent
        /// Set parent of each arg to self
        for arg in arguments {
            arg.parent = self
        }
    }

    internal convenience init(asEntryOf parent: Function) {
        self.init(name: "entry", arguments: parent.arguments, parent: parent)
    }

    public convenience init(name: String, arguments: [(String, Type)], parent: Function) {
        self.init(name: name, arguments: arguments.map(Argument.init), parent: parent)
    }

}

// MARK: - Predicates and accessors
public extension BasicBlock {

    /// Whether there exists a terminator instruction
    /// - Note: a branching instruction in the middle of the basic block
    /// is not considered a terminator
    var hasTerminator: Bool {
        return elements.last?.kind.isTerminator ?? false
    }

    /// Terminator instruction
    var terminator: Instruction? {
        guard let last = elements.last, last.kind.isTerminator else {
            return nil
        }
        return last
    }

    var successorCount: Int {
        return terminator?.controlFlowSuccessorCount ?? 0
    }

    var hasSuccessors: Bool {
        return successorCount > 0
    }

    var isReturn: Bool {
        return terminator?.kind.isReturn ?? false
    }

    var module: Module {
        return parent.parent
    }

    var isEntry: Bool {
        return parent.entry === self
    }

}

// MARK: - Arguments
public extension BasicBlock {

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
