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

public protocol IRUnit : class, HashableByReference {
    var analysisManager: AnalysisManager<Self> { get }
    var transformManager: TransformManager<Self> { get }
    func invalidateAnalyses()
}

public protocol IRSubUnit : IRUnit {
    associatedtype Parent : IRCollection, IRUnit
    unowned var parent: Parent { get set }
}
