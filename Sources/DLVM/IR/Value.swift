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

/// Scope of value
public enum Scope {
    case global
    case local
    case none
}

/// Value base
public protocol Value : SelfVerifiable {
    var type: Type { get }
    func makeUse() -> Use
}

public protocol SimpleValue : Value {
    var shape: TensorShape { get }
    var dataType: DataType { get }
}

public extension SimpleValue {
    var type: Type {
        return .tensor(shape, dataType)
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

public protocol Definition : class, Value {
}

/// User, anything that can use a value
public protocol User {
    var operands: [Use] { get }
}
