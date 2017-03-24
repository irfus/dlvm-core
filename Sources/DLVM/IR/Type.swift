//
// Created by Richard Wei on 3/10/17.
//

import DLVMTensor

public indirect enum Type {
    /// Tensor represents all scalars, vectors, matrices and higher
    /// dimensional matrices of primitive data type
    case tensor(TensorShape, DataType)
    case array(Type, Int)
    /// N-ary tuple. Corresponding to LLVM-style struct type
    case tuple([Type])
    case pointer(Type)
    case function([Type], Type)
    case alias(TypeAlias)
    case void
    case invalid
}

// MARK: - Predicates
public extension Type {
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
}

// MARK: - Subtype extraction
public extension Type {
    func subtype(at indices: [Int]) -> Type? {
        guard let idx = indices.first else { return self }
        let result: Type
        switch unaliased {
        case let .array(t, n) where idx < n:
            result = t
        case let .tuple(tt) where tt.indices.contains(idx):
            result = tt[idx]
        case let .tensor(shape, dt) where shape.rank > 0 && idx < shape[0]:
            result = .tensor(shape.dropFirst(), dt)
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
        case let .pointer(t): return .pointer(t.canonical)
        case let .function(tt, t): return.function(tt.map{$0.canonical}, t.canonical)
        case let .alias(.transparent(_, subT)): return subT.canonical
        case .alias(.opaque): return self
        case .tensor, .void, .invalid: return self
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
        case let (.array(t1, n1), .array(t2, n2)):
            return t1 == t2 && n1 == n2
        case let (.pointer(t1), .pointer(t2)):
            return t1 == t2
        case let (.function(tt1, t1), .function(tt2, t2)):
            return tt1 == tt2 && t1 == t2
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
public extension Type {
    public var isValid: Bool {
        switch self {
        case .invalid:
            return false
        case .tensor, .void:
            return true
        case let .array(subtype, _),
             let .pointer(subtype):
            return subtype.isValid
        case let .tuple(subtypes):
            return subtypes.reduce(true, { $0 && $1.isValid })
        case let .function(args, ret):
            return args.reduce(true, { $0 && $1.isValid }) && ret.isValid
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
