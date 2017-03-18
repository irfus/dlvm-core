//
//  IRObject.swift
//  DLVM
//
//  Created by Richard Wei on 12/25/16.
//
//

public protocol EquatableByReference : class, Equatable {}
public protocol HashableByReference : EquatableByReference, Hashable {}

public extension EquatableByReference {
    public static func == (lhs: Self, rhs: Self) -> Bool {
        return lhs === rhs
    }
}

public extension HashableByReference {
    public var hashValue: Int {
        return ObjectIdentifier(self).hashValue
    }
}

public protocol IRUnit : class, HashableByReference, SelfVerifiable {
    var analysisManager: AnalysisManager<Self> { get }
    var transformManager: TransformManager<Self> { get }
    func invalidateAnalyses()
}

public protocol IRSubUnit : IRUnit {
    associatedtype Parent : IRCollection, IRUnit
    unowned var parent: Parent { get set }
}

public extension IRSubUnit
    where Parent : IRCollection,
          Parent.ElementCollection : OrderedMapSetProtocol,
          Parent.ElementCollection.Iterator.Element == Self,
          Parent.ElementCollection.Element == Parent.ElementCollection.Iterator.Element
{
    var indexInParent: Int {
        guard let index = parent.index(of: self) else {
            preconditionFailure("Self does not exist in parent basic block")
        }
        return index
    }

    func removeFromParent() {
        parent.remove(self)
    }
}
