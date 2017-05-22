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

public final class Function : Named, IRCollection, IRUnit {
    public enum Attribute {
        case differentiable
        case inline
        case differentiating(Function, from: Int, wrt: [Int], keepingOutputs: [Int])
    }

    public typealias Element = BasicBlock

    public var name: String
    public var result: Type
    public var arguments: OrderedSet<Argument> = []
    public var attributes: Set<Attribute> = []
    public unowned var parent: Module

    public var elements: OrderedSet<BasicBlock> = []
    public internal(set) var analysisManager: AnalysisManager<Function> = AnalysisManager()

    public init(name: String, arguments: [(String, Type)],
                result: Type, attributes: Set<Attribute>, parent: Module) {
        self.name = name
        self.arguments.append(contentsOf: arguments.map { Argument(name: $0.0, type: $0.1) })
        self.result = result
        self.attributes = attributes
        self.parent = parent
    }
}

// MARK: - Hashable
extension Function.Attribute : Hashable {
    public static func == (lhs: Function.Attribute, rhs: Function.Attribute) -> Bool {
        switch (lhs, rhs) {
        /// Equality by case handle
        case (.differentiable, .differentiable),
             (.inline, .inline),
             (.differentiating, .differentiating):
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
        }
    }
}

// MARK: - Attribute helper
public extension Function {
    var isDifferentiable: Bool {
        return attributes.contains(.differentiable)
    }
}

// MARK: - Value
extension Function : Value {
    public var type: Type {
        return .function(arguments.map{$0.type}, result)
    }

    public func makeUse() -> Use {
        return .function(type, self)
    }
}

// MARK: - Arguments
public extension Function {

    func acceptsArguments<C : Collection>(_ types: C) -> Bool where C.Iterator.Element == Type, C.IndexDistance == Int {
        guard types.count == arguments.count else { return false }
        return zip(types, arguments).forAll{$0.0.conforms(to: $0.1.type)}
    }

}

// MARK: - Control flow
public extension Function {

    var instructions: LazyCollection<FlattenBidirectionalCollection<Function>> {
        return lazy.joined()
    }

}

// MARK: - Gradient information
public extension Function {

    func gradientType(fromOutput diffIndex: Int,
                      withRespectTo varIndices: [Int],
                      keepingOutputs outputIndices: [Int]) -> Type? {
        var keptOutputs: [Type]
        /// Check output index
        switch result {
        /// Tuple output is treated as multiplt-out
        case let .tuple(subtypes):
            guard
                /// Diff index must be in bounds
                subtypes.indices.contains(diffIndex),
                /// Indices of the outputs to keep must be in bounds
                let someOutputs = subtypes.subcollection(atIndices: outputIndices),
                /// Indices of the outputs to keep must not contain any duplicate
                !outputIndices.containsDuplicate
                else { return nil }
            keptOutputs = someOutputs
        /// Other output is treated as single out
        case _ where
                /// Result must be differentiable
                result.isDifferentiable
                /// Index must be 0
                && diffIndex == 0
                /// Indices of the outputs to keep must be either [] or [0] 
                && (outputIndices.isEmpty || outputIndices == [0]):
            keptOutputs = outputIndices.isEmpty ? [] : [result]
        /// Bad differentiation case
        default:
            return nil
        }
        /// Check variable indices
        let argTypes = arguments.map{$0.type}
        guard
            /// Indices of diff variables must be in bounds
            let diffVars = argTypes.subcollection(atIndices: varIndices),
            /// All diff variables must be diff'able arguments
            diffVars.forAll({$0.isDifferentiable})
            else { return nil }
        /// Result of differentiation has the same input types but different output types 
        /// Output type is `(k1, ..., kn, d1, ..., dn)` where `k1...kn` are outputs of the 
        /// original function to keep, `d1...dn` are derivatives of the output at `diffIndex`
        /// with respect to arguments at indices `varIndices`, respectively.
        return .function(argTypes, .tuple(keptOutputs + diffVars))
    }
    
}
