//
//  Ownership.swift
//  DLVM
//
//  Created by Richard Wei on 2/19/17.
//
//

/// "deref" operator that looks like C's deref
/// This is a hacky operator which is equivalent to
/// Unowned.object / Weak.object
/// - Note: With Swift 4's memory ownership model,
/// we'll no longer need Unowned/Weak wrappers. Yay!
prefix operator *

public struct Box<Object: AnyObject> {
    var object: Object

    init(_ object: Object) {
        self.object = object
    }

    static prefix func * (ref: Box) -> Object {
        return ref.object
    }
}

struct Unowned<Object: AnyObject> {
    unowned var object: Object

    init(_ object: Object) {
        self.object = object
    }

    static prefix func * (ref: Unowned) -> Object {
        return ref.object
    }
}

struct Weak<Object: AnyObject> {
    weak var object: Object?

    init(_ object: Object) {
        self.object = object
    }

    static prefix func * (ref: Weak) -> Object? {
        return ref.object
    }
}

extension Box : Hashable {

    public static func ==(lhs: Box, rhs: Box) -> Bool {
        return lhs.object === rhs.object
    }

    public var hashValue: Int {
        return ObjectIdentifier(object).hashValue
    }

}

extension Unowned : Hashable {

    static func ==(lhs: Unowned, rhs: Unowned) -> Bool {
        return lhs.object === rhs.object
    }

    var hashValue: Int {
        return ObjectIdentifier(object).hashValue
    }
    
}

extension Weak : Hashable {

    static func ==(lhs: Weak, rhs: Weak) -> Bool {
        return lhs.object === rhs.object
    }

    var hashValue: Int {
        return object.flatMap { ObjectIdentifier($0).hashValue } ?? 0
    }
    
}

