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

public protocol IRObject : class, HashableByReference {
    associatedtype Parent : AnyObject
    weak var parent: Parent? { get }
}