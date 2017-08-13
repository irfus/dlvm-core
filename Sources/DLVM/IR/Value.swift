//
//  Value.swift
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

import CoreTensor

/// Value base
public protocol Value : Verifiable {
    var type: Type { get }
    func makeUse() -> Use
}

/// % operator turns a def to a use
prefix operator %

public extension Value {
    prefix static func % (value: Self) -> Use {
        return value.makeUse()
    }

    prefix static func % (value: Value) -> Use {
        return value.makeUse()
    }
}

/// Anything that has a name
public protocol Named {
    var name: String { get }
}

/// Anything that may have a name
public protocol MaybeNamed {
    var name: String? { get }
}

/// User, anything that can use a value
public protocol User {
    var operands: [Use] { get }
}

public typealias Definition = Value & AnyObject
