//
//  Type.swift
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

/// Element key to form a key path in GEP and use
public enum ElementKey {
    case index(Int)
    case name(String)
    case value(Use)
}

// MARK: - Equatable
extension ElementKey : Equatable {
    public static func == (lhs: ElementKey, rhs: ElementKey) -> Bool {
        switch (lhs, rhs) {
        case let (.index(i1), .index(i2)): return i1 == i2
        case let (.name(n1), .name(n2)): return n1 == n2
        case let (.value(v1), .value(v2)): return v1 == v2
        default: return false
        }
    }
}

/// Nominal type
public class StructType : Named, HashableByReference {
    public enum Attribute {
        case packed
    }
    public typealias Field = (name: String, type: Type)
    public var name: String
    public var fields: [Field]
    public var attributes: Set<Attribute> = []

    public init(name: String, fields: [Field], attributes: Set<Attribute>) {
        self.name = name
        self.fields = fields
        self.attributes = attributes
    }
}

/// - TODO: add enum type

// MARK: - Accessors
public extension StructType {
    var type: Type {
        return .struct(self)
    }
    
    var isPacked: Bool {
        return attributes.contains(.packed)
    }
    
    func field(named name: String) -> Field? {
        return fields.first(where: {$0.name == name})
    }

    func subtype(named name: String) -> Type? {
        return field(named: name)?.type
    }

    var subtypes: [Type] {
        return fields.map {$0.1}
    }

    func subtype(at key: ElementKey) -> Type? {
        guard case .name(let name) = key else { return nil }
        return subtype(named: name)
    }

    func indexOfField(named name: String) -> Int? {
        return fields.index(where: { $0.name == name })
    }
}

/// Type alias
public class TypeAlias : Named, HashableByReference {
    public var name: String
    public var type: Type?

    public init(name: String, type: Type? = nil) {
        self.name = name
        self.type = type
    }

    public var isOpaque: Bool {
        return type == nil
    }
}

/// Type of IR values
public indirect enum Type {
    /// Tensor represents all scalars, vectors, matrices and higher
    /// dimensional matrices of primitive data type
    /// - Note: A tensor is transparent to LLVM as a multi-dimensional array,
    /// or in some cases a vector. LLGen target decides this.
    case tensor(TensorShape, DataType)
    /// Fixed sized array
    case array(Int, Type)
    /// N-ary tuple, corresponding to LLVM unpacked struct type
    case tuple([Type])
    /// Struct, corresponding to LLVM struct type
    case `struct`(StructType)
    /// Pointer
    case pointer(Type)
    /// Reference counted box
    case box(Type)
    /// Function type
    case function([Type], Type)
    /// Alias type, transparent or opaque
    case alias(TypeAlias)
    /// Doesn't contain any value. No size
    case void
    /// Invalid type during type inference, to be eliminated by
    /// the verifier
    case invalid
}

// MARK: - Factories
public extension Type {
    static func int(_ size: UInt) -> Type {
        return .scalar(.int(size))
    }

    static func float(_ size: FloatingPointSize) -> Type {
        return .scalar(.float(size))
    }

    static func bool() -> Type {
        return .scalar(.bool)
    }
    
    static func scalar(_ dataType: DataType) -> Type {
        return .tensor(.scalar, dataType)
    }

    var pointer: Type {
        return .pointer(self)
    }
}

// MARK: - Properties
public extension Type {
    enum BitSize {
        case number(Int)
        case pointer
    }

    var isPassedAsPointer: Bool {
        switch canonical {
        case .pointer, .function: return true
        default: return false
        }
    }

    var isNotSized: Bool {
        switch canonical {
        case .invalid, .void: return true
        default: return false
        }
    }
}

// MARK: - Predicates
public extension Type {
    var isFirstClass: Bool {
        switch canonical {
        case .tensor, .array, .tuple, .pointer, .box, .alias: return true
        default: return false
        }
    }
    
    var isTensor: Bool {
        switch canonical {
        case .tensor: return true
        default: return false
        }
    }

    var isScalar: Bool {
        switch canonical {
        case .tensor([], _): return true
        default: return false
        }
    }

    var isVoid: Bool {
        switch canonical {
        case .void: return true
        default: return false
        }
    }

    func conforms(to other: Type) -> Bool {
        return canonical == other.canonical
    }
}

// MARK: - Subtype extraction
public extension Type {
    func subtype(at key: ElementKey) -> Type? {
        switch (unaliased, key) {
        case let (.tuple(tt), .index(i)) where tt.indices.contains(i):
            return tt[i]
        case let (.struct(structTy), .name(_)):
            guard let ty = structTy.subtype(at: key) else { return nil }
            return ty
        case let (.tensor(shape, dt), .index(i)) where shape.rank > 0 && i < shape[0]:
            return .tensor(shape.dropFirst(), dt)
        case let (.array(n, t), .index(i)) where i < n:
            return t
        case let (.pointer(t), .index(_)):
            return t
        default:
            return nil
        }
    }
    
    func subtype(at keys: [ElementKey]) -> Type? {
        return keys.reduce(self, { acc, key in
            acc.flatMap { $0.subtype(at: key) }
        })
    }
}

// MARK: - Alias resolution and canonicalization
public extension Type {
    var canonical: Type {
        switch self {
        case let .array(i, subT): return .array(i, subT.canonical)
        case let .tuple(tt): return .tuple(tt.map{$0.canonical})
        case let .pointer(t): return .pointer(t.canonical)
        case let .box(t): return .box(t.canonical)
        case let .function(tt, t): return.function(tt.map{$0.canonical}, t.canonical)
        case let .alias(alias): return alias.type?.canonical ?? self
        case .tensor, .void, .invalid, .struct: return self
        }
    }

    var unaliased: Type {
        if case let .alias(alias) = self {
            return alias.type?.unaliased ?? self
        }
        return self
    }

    var isDifferentiable: Bool {
        switch self {
        case let .tensor([], dt),
             let .tensor(_, dt) where dt.isNumeric:
            return true
        default:
            return false
        }
    }
}

// MARK: - Equatable
extension Type : Equatable {
    public static func ==(lhs: Type, rhs: Type) -> Bool {
        switch (lhs.canonical, rhs.canonical) {
        case let (.tensor(s1, t1), .tensor(s2, t2)):
            return s1 == s2 && t1 == t2
        case let (.tuple(ts1), .tuple(ts2)):
            return ts1 == ts2
        case let (.struct(s1), .struct(s2)):
            return s1 === s2
        case let (.array(t1, n1), .array(t2, n2)):
            return t1 == t2 && n1 == n2
        case let (.pointer(t1), .pointer(t2)):
            return t1 == t2
        case let (.box(t1), .box(t2)):
            return t1 == t2
        case let (.function(tt1, t1), .function(tt2, t2)):
            return tt1 == tt2 && t1 == t2
        case (.void, .void), (.invalid, .invalid):
            return true
        case let (.alias(a1), .alias(a2)) where a1.isOpaque && a2.isOpaque:
            return a1.name == a2.name
        default:
            return false
        }
    }
}

// MARK: - Validation
public extension StructType {
    var isValid: Bool {
        return subtypes.forAll { $0.isValid }
    }
}

// MARK: - Validation
public extension Type {
    public var isValid: Bool {
        switch self {
        case .invalid:
            return false
        case .tensor, .void:
            return true
        case let .array(_, subtype),
             let .pointer(subtype),
             let .box(subtype):
            return subtype.isValid
        case let .tuple(subtypes):
            return subtypes.forAll { $0.isValid }
        case let .struct(structTy):
            return structTy.isValid
        case let .function(args, ret):
            return args.forAll { $0.isValid } && ret.isValid
        case let .alias(a):
            return a.isValid
        }
    }
}

// MARK: - Validation
public extension TypeAlias {
    public var isValid: Bool {
        return type?.isValid ?? true
    }
}

// MARK: - Literal helpers
public extension Type {
    public func makeZero() -> LiteralValue {
        return LiteralValue(type: self, literal: .zero)
    }
}
