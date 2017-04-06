//
//  Ownership.swift
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

/// "deref" operator that looks like C's deref
/// This is a hacky operator which is equivalent to
/// Unowned.object / Weak.object
/// - Note: With Swift 4's memory ownership model,
/// we'll no longer need Unowned/Weak wrappers. Yay!
prefix operator *

public struct Owned<Object: AnyObject> {
    var object: Object

    init(_ object: Object) {
        self.object = object
    }

    static prefix func * (ref: Owned) -> Object {
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

extension Owned: Hashable {

    public static func ==(lhs: Owned, rhs: Owned) -> Bool {
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

