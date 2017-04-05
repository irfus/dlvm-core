//
// Created by Richard Wei on 3/10/17.
//

import DLVMTensor

public enum Mutability {
    case mutable, immutable
}

public struct StructType {
    public typealias Field = (name: String, type: Type)
    public var fields: [(name: String, type: Type)]
    public var isPacked: Bool

    public init(fields: [Field], isPacked: Bool = false) {
        self.fields = fields
        self.isPacked = isPacked
    }
}

extension StructType : Equatable {
    public static func == (lhs: StructType, rhs: StructType) -> Bool {
        return lhs.fields.elementsEqual(rhs.fields, by: { $0.name == $1.name && $0.type == $1.type })
            && lhs.isPacked == rhs.isPacked
    }
}

public extension StructType {
    func field(named name: String) -> Field? {
        return fields.first(where: {$0.name == name})
    }

    var subtypes: [Type] {
        return fields.map {$1}
    }

    var canonical: StructType {
        return StructType(fields: fields.map{($0, $1.canonical)}, isPacked: isPacked)
    }
}

public indirect enum Type {
    /// Tensor represents all scalars, vectors, matrices and higher
    /// dimensional matrices of primitive data type
    /// - Note: A tensor is transparent to LLVM as a multi-dimensional array,
    /// or in some cases a vector. LLGen target decides this.
    case tensor(TensorShape, DataType)
    /// Fixed sized array
    case array(Type, Int)
    /// N-ary tuple, corresponding to LLVM unpacked struct type
    case tuple([Type])
    /// Struct, corresponding to LLVM struct type
    case `struct`(StructType)
    /// Pointer
    case pointer(Type, MemoryLocation, Mutability)
    /// Reference counted box
    case box(Type, MemoryLocation)
    /// Function type
    case function([Type], Type)
    /// Alias type, transparent or opaque
    case alias(TypeAlias)
    /// Compute graph typed by a function reference
    /// - Note: this is a special reference type that will
    /// be lowered to a pointer to a struct containing allocated
    /// graph nodes for compute
    case computeGraph(Function)
    /// Doesn't contain any value. No size
    case void
    /// Invalid type during type inference, to be eliminated by
    /// the verifier
    case invalid
}

// MARK: - Factories
public extension Type {
    static func scalar(_ dataType: DataType) -> Type {
        return .tensor(.scalar, dataType)
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
        case .tensor, .array, .tuple, .pointer, .box, .computeGraph, .alias: return true
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
        switch (self.canonical, other.canonical) {
        case let (.pointer(t1, loc1, .mutable), .pointer(t2, loc2, .immutable)):
            return t1 == t2 && loc1 == loc2
        default:
            return self == other
        }
    }
}

// MARK: - Subtype extraction
public extension Type {
    func subtype(at indices: [Int]) -> Type? {
        guard let idx = indices.first else { return self }
        let result: Type
        switch unaliased {
        case let .tuple(tt) where tt.indices.contains(idx):
            result = tt[idx]
        case let .struct(structTy) where structTy.fields.indices.contains(idx):
            result = structTy.fields[idx].type
        case let .tensor(shape, dt) where shape.rank > 0 && idx < shape[0]:
            result = .tensor(shape.dropFirst(), dt)
        case let .array(t, n) where idx < n:
            result = t
        case let .pointer(t, _, _), let .box(t, _):
            result = t
        default:
            return nil
        }
        return result.subtype(at: Array(indices.dropFirst()))
    }
}

// MARK: - Alias resolution and canonicalization
public extension Type {
    var canonical: Type {
        switch self {
        case let .array(subT, i): return .array(subT.canonical, i)
        case let .tuple(tt): return .tuple(tt.map{$0.canonical})
        case let .struct(ty): return .struct(ty.canonical)
        case let .pointer(t, loc, mut): return .pointer(t.canonical, loc, mut)
        case let .box(t, loc): return .box(t.canonical, loc)
        case let .function(tt, t): return.function(tt.map{$0.canonical}, t.canonical)
        case let .alias(.transparent(_, subT)): return subT.canonical
        case .alias(.opaque): return self
        case .tensor, .void, .invalid, .computeGraph: return self
        }
    }

    var unaliased: Type {
        switch self {
        case let .alias(.transparent(_, t)): return t.unaliased
        default: return self
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
            return s1 == s2
        case let (.array(t1, n1), .array(t2, n2)):
            return t1 == t2 && n1 == n2
        case let (.pointer(t1, loc1, mut1), .pointer(t2, loc2, mut2)):
            return t1 == t2 && loc1 == loc2 && mut1 == mut2
        case let (.box(t1, loc1), .box(t2, loc2)):
            return t1 == t2 && loc1 == loc2
        case let (.function(tt1, t1), .function(tt2, t2)):
            return tt1 == tt2 && t1 == t2
        case let (.computeGraph(f1), .computeGraph(f2)):
            return f1 === f2
        case (.void, .void), (.invalid, .invalid):
            return true
        case let (.alias(.opaque(name1)), .alias(.opaque(name2))):
            return name1 == name2
        default:
            return false
        }
    }
}

// MARK: - Validation
public extension StructType {
    var isValid: Bool {
        return fields.forAll { $1.isValid }
    }
}

// MARK: - Validation
public extension Type {
    public var isValid: Bool {
        switch self {
        case .invalid:
            return false
        case .tensor, .void, .computeGraph:
            return true
        case let .array(subtype, _),
             let .pointer(subtype, _, _),
             let .box(subtype, _):
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
        switch self {
        case .opaque:
            return true
        case let .transparent(_, t):
            return t.isValid
        }
    }
}

// MARK: - Literal helpers
public extension Type {
    public func makeZero() -> LiteralValue {
        return LiteralValue(type: self, literal: .zero)
    }
}
