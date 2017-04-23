//
//  Intrinsics.swift
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

public final class Function : Named, IRCollection, IRSubUnit {
    public enum Attribute {
        case differentiable
        case inline
        case differentiating(Function, from: Int, wrt: [Int], keepingOutputs: [Int])
        case compute
    }

    public typealias Element = BasicBlock

    public var name: String
    public var result: Type
    public var arguments: OrderedMapSet<Argument> = []
    public var attributes: Set<Attribute> = []
    public unowned var parent: Module

    public var elements: OrderedMapSet<BasicBlock> = []
    public internal(set) var analysisManager: AnalysisManager<Function> = AnalysisManager()
    public internal(set) var transformManager: TransformManager<Function> = TransformManager()

    public unowned var entry: BasicBlock {
        if let entry = elements["entry"] {
            return entry
        }
        let bb = BasicBlock(asEntryOf: self)
        elements.append(bb)
        return bb
    }

    public init(name: String, arguments: [(String, Type)],
                result: Type, attributes: Set<Attribute>, parent: Module) {
        self.name = name
        self.arguments.append(contentsOf: arguments.map(Argument.init))
        self.result = result
        self.attributes = attributes
        self.parent = parent
        _ = entry
    }
}

// MARK: - Hashable
extension Function.Attribute : Hashable {
    public static func == (lhs: Function.Attribute, rhs: Function.Attribute) -> Bool {
        switch (lhs, rhs) {
        /// Equality by case handle
        case (.differentiable, .differentiable),
             (.inline, .inline),
             (.differentiating, .differentiating),
             (.compute, .compute):
            return true
        default:
            return false
        }
    }
    
    public var hashValue: Int {
        switch self {
        case .differentiable:  return 1
        case .inline:          return 2
        case .differentiating: return 3
        case .compute:         return 4
        }
    }
}

// MARK: - Attribute helper
public extension Function {
    var isDifferentiable: Bool {
        return attributes.contains(.differentiable)
    }

    var isCompute: Bool {
        return attributes.contains(.compute)
    }
}

// MARK: - Value
extension Function : Value, Definition {
    public var type: Type {
        return .function(arguments.map{$0.type}, result)
    }

    public func makeUse() -> Use {
        return .function(self)
    }
}

// MARK: - Arguments
public extension Function {

    func acceptsArguments<C : Collection>(_ types: C) -> Bool where C.Iterator.Element == Type, C.IndexDistance == Int {
        guard types.count == arguments.count else { return false }
        return zip(types, arguments).forAll{$0.conforms(to: $1.type)}
    }

    func argument(named name: String) -> Argument? {
        return arguments.element(named: name)
    }

    func argumentValue(named name: String) -> Use? {
        return argument(named: name).flatMap { .argument($0.type, $0) }
    }

    func containsArgument(named name: String) -> Bool {
        return arguments.containsElement(named: name)
    }
    
}

// MARK: - Control flow
public extension Function {

    var instructions: LazyCollection<FlattenBidirectionalCollection<Function>> {
        return lazy.joined()
    }

    func instruction(named name: String) -> Instruction? {
        for bb in self {
            if let inst = bb.element(named: name) {
                return inst
            }
        }
        return nil
    }

    func containsInstruction(named name: String) -> Bool {
        return instruction(named: name) != nil
    }

    func containsName(_ name: String) -> Bool {
        return containsElement(named: name) || contains(where: {
            $0.containsArgument(named: name) || $0.containsElement(named: name)
        })
    }

}

// MARK: - Gradient information
public extension Function {

    func gradientType(fromOutput diffIndex: Int,
                      withRespectTo varIndices: [Int],
                      keepingOutputs outputIndices: [Int]) -> Type? {
        /// Check output index
        switch result {
        case let .tuple(subtypes):
            guard
                /// Diff index must be in bounds
                subtypes.indices.contains(diffIndex),
                /// Indices of the outputs to keep must be in bounds
                let _ = subtypes.subcollection(atIndices: outputIndices),
                /// Indices of the outputs to keep must not contain any duplicate
                !outputIndices.containsDuplicate
                else { return nil }
        case let type where
                /// Single result must be diff'able
                !type.isDifferentiable
                /// Diff index must be 0
                || diffIndex != 0
                /// Kept output indices must be [0]
                || outputIndices != [0]:
            return nil
        default:
            break
        }
        /// Check variable indices
        let argTypes = arguments.map{$0.type}
        guard
            /// Indices of diff variables must be in bounds
            let diffVars = argTypes.subcollection(atIndices: varIndices),
            /// All diff variables must be diff'able arguments
            diffVars.forAll({$0.isDifferentiable})
            else { return nil }
        return .function(argTypes, .tuple(diffVars))
    }
    
}
