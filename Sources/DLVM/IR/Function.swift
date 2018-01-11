//
//  Intrinsics.swift
//  DLVM
//
//  Copyright 2016-2018 The DLVM Team.
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

public struct GradientConfiguration : Equatable {
    /// Function to differentiate.
    let primal: Function
    /// Index of tuple element to differentiate, when the return type is a
    /// tuple; otherwise must be 0.
    let sourceIndex: Int?
    /// Indices of arguments to differentiate the function with respect to.
    let argumentIndices: [Int]?
    /// Indices of return values to kept in the gradient function, when the
    /// return type is a tuple; otherwise can be [0] or [].
    let keptIndices: [Int]
    /// Whether the gradient function can take a back-propagated gradient as the
    /// seed for reverse-mode AD.
    let isSeedable: Bool

    /// Create a gradient configuration.
    /// - Note: The type of the forward function must be consistent with the
    ///         configuration.
    public init(primal: Function, sourceIndex: Int?, argumentIndices: [Int]?,
                keptIndices: [Int], isSeedable: Bool) {
        self.primal = primal
        self.sourceIndex = sourceIndex
        self.argumentIndices = argumentIndices
        self.keptIndices = keptIndices
        self.isSeedable = isSeedable
    }
}

public final class Function : Named, IRCollection, IRUnit {
    public enum Attribute {
        /// Inline the function during LLGen
        case inline
    }

    public enum DeclarationKind {
        /// Externally defined
        case external
        /// Marks a function as the gradient of another with given
        /// configuration. To be materialized as a normally defined function by
        /// AD.
        case gradient(GradientConfiguration)
    }

    public typealias Base = OrderedSet<BasicBlock>
    public typealias Element = BasicBlock

    public var name: String
    public var argumentTypes: [Type]
    public var returnType: Type
    public var attributes: Set<Attribute> = []
    public var declarationKind: DeclarationKind?
    public var parent: Module

    public var elements: OrderedSet<BasicBlock> = []
    public internal(set) var passManager: PassManager<Function> = PassManager()

    public init(name: String, argumentTypes: [Type],
                returnType: Type, attributes: Set<Attribute> = [],
                declarationKind: DeclarationKind? = nil, parent: Module) {
        self.name = name
        self.argumentTypes = argumentTypes
        self.returnType = returnType
        self.attributes = attributes
        self.declarationKind = declarationKind
        self.parent = parent
    }

    public var canApplyTransforms: Bool {
        return !isDeclaration
    }
}

/// - Note: This is a workaround for a type checker bug in Swift 4
public extension Function {
    func append(_ newElement: Element) {
        elements.append(newElement)
        newElement.parent = self
        invalidatePassResults()
    }

    func insert(_ newElement: Element, at index: Base.Index) {
        elements.insert(newElement, at: index)
        newElement.parent = self
        invalidatePassResults()
    }

    func insert(_ newElement: Element, after other: Element) {
        elements.insert(newElement, after: other)
        newElement.parent = self
        invalidatePassResults()
    }

    func insert(_ newElement: Element, before other: Element) {
        elements.insert(newElement, before: other)
        newElement.parent = self
        invalidatePassResults()
    }
}

extension Function : Value {
    public var type: Type {
        return .function(argumentTypes, returnType)
    }

    public func makeUse() -> Use {
        return .function(type, self)
    }
}

public extension Function {
    func acceptsArguments<C : Collection>(_ types: C) -> Bool
        where C.Element == Type
    {
        guard types.count == argumentTypes.count else { return false }
        return zip(types, argumentTypes).forAll { actual, formal in
            actual.conforms(to: formal)
        }
    }

    var isDeclaration: Bool {
        return declarationKind != nil
    }

    var isDefinition: Bool {
        return declarationKind == nil
    }

    var instructions: LazyCollection<FlattenCollection<Function>> {
        return lazy.joined()
    }
}

public extension Function {
    /// Replace all occurrences of an instruction with another use
    func replaceAllUses(of oldInstruction: Instruction, with newUse: Use) {
        /// If `instruction` exists in its parent BB in this function,
        /// we only search the area after `instruction`'s definition
        if oldInstruction.existsInParent, oldInstruction.parent.parent == self {
            let bbIndex = oldInstruction.parent.indexInParent
            let instIndex = oldInstruction.indexInParent
            for bb in suffix(from: bbIndex) {
                for inst in bb where bb != oldInstruction.parent ||
                    inst.indexInParent >= instIndex {
                    inst.substitute(newUse, for: %oldInstruction)
                }
            }
        }
        /// Otherwise, we search every use for `instruction`
        else {
            for inst in instructions {
                inst.substitute(newUse, for: %oldInstruction)
            }
        }
    }

    /// Replace all occurrences of a use with another use
    func replaceAllUses(of oldUse: Use, with newUse: Use) {
        switch oldUse {
        case let .instruction(_, inst):
            replaceAllUses(of: inst, with: newUse)
        default:
            for inst in instructions {
                inst.substitute(newUse, for: oldUse)
            }
        }
    }
}

public extension Function {
    func gradientType(from sourceIndex: Int?,
                      wrt argIndices: [Int]?,
                      keeping keptIndices: [Int],
                      seedable isSeedable: Bool) -> Type? {
        var keptOutputs: [Type]
        let sourceType: Type
        /// Check output index
        switch returnType {
        /// Tuple output is treated as multiple-out
        case let .tuple(elementTypes):
            guard let sourceIndex = sourceIndex,
                /// Source index must be in bounds
                elementTypes.indices.contains(sourceIndex),
                /// Element must be differentiable
                elementTypes[sourceIndex].isDifferentiable,
                /// Indices of the outputs to keep must be in bounds
                let someOutputs = elementTypes
                    .subcollection(atIndices: keptIndices),
                /// Indices of the outputs to keep must not contain any
                /// duplicate
                !keptIndices.containsDuplicate
                else { return nil }
            sourceType = elementTypes[sourceIndex]
            keptOutputs = someOutputs
        /// Other output is treated as single-out
        case _ where sourceIndex == nil
                /// Result must be differentiable
                && returnType.isDifferentiable
                /// Indices of the outputs to keep must be either [] or [0]
                && (keptIndices.isEmpty || keptIndices == [0]):
            sourceType = returnType
            keptOutputs = keptIndices.isEmpty ? [] : [returnType]
        /// Bad differentiation case
        default:
            return nil
        }
        /// Check arguments
        let argIndices = argIndices ?? Array(argumentTypes.indices)
        guard
            /// Indices of diff variables must be in bounds
            let diffVars = argumentTypes.subcollection(atIndices: argIndices),
            /// All diff variables must be diff'able arguments
            diffVars.forAll({ $0.isDifferentiable })
            else { return nil }
        /// Result of differentiation has the same input types but different
        /// output types Output type is `(k1, ..., kn, d1, ..., dn)` where
        /// `k1...kn` are outputs of the original function to keep, `d1...dn`
        /// are derivatives of the output at `sourceIndex` with respect to
        /// arguments at indices `argIndices`, respectively.
        return .function(
            isSeedable ? argumentTypes + [sourceType] : argumentTypes,
            .tuple(diffVars + keptOutputs)
        )
    }
}
