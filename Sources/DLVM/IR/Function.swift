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
        /// Inline the function during LLGen
        case inline
        /// Mark the function as the gradient of another with given configuration
        /// - Parameters:
        ///   - from: index of tuple element to differentiate, when the return type is
        ///           a tuple; otherwise must be 0
        ///   - wrt: indices of arguments to differentiate the function with respect to
        ///   - keeping: indices of return values to kept in the gradient function, when
        ///              the return type is a tuple; otherwise can be [0] or []
        /// - Note: The type of the function must match that given by the configuration
        case gradient(Function, from: Int, wrt: [Int], keeping: [Int], seedable: Bool)
    }

    public typealias Element = BasicBlock

    public var name: String
    public var argumentTypes: [Type]
    public var returnType: Type
    public var attributes: Set<Attribute> = []
    public unowned var parent: Module

    public var elements: OrderedSet<BasicBlock> = []
    public internal(set) var analysisManager: AnalysisManager<Function> = AnalysisManager()

    public init(name: String, argumentTypes: [Type],
                returnType: Type, attributes: Set<Attribute>, parent: Module) {
        self.name = name
        self.argumentTypes = argumentTypes
        self.returnType = returnType
        self.attributes = attributes
        self.parent = parent
    }
}

// MARK: - Hashable
extension Function.Attribute : Hashable {
    public static func == (lhs: Function.Attribute, rhs: Function.Attribute) -> Bool {
        switch (lhs, rhs) {
        /// Equality by case handle
        case (.inline, .inline),
             (.gradient, .gradient):
            return true
        default:
            return false
        }
    }
    
    public var hashValue: Int {
        switch self {
        case .inline:          return 0
        case .gradient: return 1
        }
    }
}

// MARK: - Value
extension Function : Value {
    public var type: Type {
        return .function(argumentTypes, returnType)
    }

    public func makeUse() -> Use {
        return .function(type, self)
    }
}

// MARK: - Arguments
public extension Function {

    func acceptsArguments<C : Collection>(_ types: C) -> Bool where C.Iterator.Element == Type, C.IndexDistance == Int {
        guard types.count == argumentTypes.count else { return false }
        return zip(types, argumentTypes).forAll { actual, formal in
            actual.conforms(to: formal)
        }
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
                      keepingOutputs outputIndices: [Int],
                      isSeedable: Bool) -> Type? {
        var keptOutputs: [Type]
        let diffSourceType: Type
        /// Check output index
        switch returnType {
        /// Tuple output is treated as multiple-out
        case let .tuple(subtypes):
            guard
                /// Diff index must be in bounds
                subtypes.indices.contains(diffIndex),
                /// Indices of the outputs to keep must be in bounds
                let someOutputs = subtypes.subcollection(atIndices: outputIndices),
                /// Indices of the outputs to keep must not contain any duplicate
                !outputIndices.containsDuplicate
                else { return nil }
            diffSourceType = subtypes[diffIndex]
            keptOutputs = someOutputs
        /// Other output is treated as single out
        case _ where
                /// Result must be differentiable
                returnType.isDifferentiable
                /// Index must be 0
                && diffIndex == 0
                /// Indices of the outputs to keep must be either [] or [0] 
                && (outputIndices.isEmpty || outputIndices == [0]):
            diffSourceType = returnType
            keptOutputs = outputIndices.isEmpty ? [] : [returnType]
        /// Bad differentiation case
        default:
            return nil
        }
        guard
            /// Indices of diff variables must be in bounds
            let diffVars = argumentTypes.subcollection(atIndices: varIndices),
            /// All diff variables must be diff'able arguments
            diffVars.forAll({$0.isDifferentiable})
            else { return nil }
        /// Result of differentiation has the same input types but different output types
        /// Output type is `(k1, ..., kn, d1, ..., dn)` where `k1...kn` are outputs of the 
        /// original function to keep, `d1...dn` are derivatives of the output at `diffIndex`
        /// with respect to arguments at indices `varIndices`, respectively.
        return .function(isSeedable ? argumentTypes + [diffSourceType] : argumentTypes,
                         .tuple(keptOutputs + diffVars))
    }
    
}
