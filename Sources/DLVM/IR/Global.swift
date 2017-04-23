//
//  Global.swift
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

/// Global value
public class GlobalValue : Value, Definition, HashableByReference {
    public enum Kind {
        case variable, constant
    }
    public var name: String
    public var kind: Kind
    public var type: Type
    public var initializer: LiteralValue

    public func makeUse() -> Use {
        return .global(type, self)
    }

    public init(name: String, kind: Kind, type: Type, initializer: LiteralValue) {
        self.name = name
        self.kind = kind
        self.type = type
        self.initializer = initializer
    }
}
