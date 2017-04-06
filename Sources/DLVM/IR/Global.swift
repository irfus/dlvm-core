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

/// Type alias
public enum TypeAlias {
    case transparent(String, Type)
    case opaque(String)
}

// MARK: - Named
extension TypeAlias : Named {
    public var name: String {
        switch self {
        case .transparent(let name, _): return name
        case .opaque(let name): return name
        }
    }
}

// MARK: - Hashable
extension TypeAlias : Hashable {
    public static func ==(lhs: TypeAlias, rhs: TypeAlias) -> Bool {
        switch (lhs, rhs) {
        case let (.transparent(n1, t1), .transparent(n2, t2)):
            return n1 == n2 && t1 == t2
        case let (.opaque(n1), .opaque(n2)):
            return n1 == n2
        default:
            return false
        }
    }

    public var hashValue: Int {
        return name.hashValue
    }
}
