//
// Created by Richard Wei on 2/24/17.
//

/// Global value
public class GlobalValue : Value, Definition {
    public enum Kind {
        case variable, constant
    }
    public var name: String
    public var kind: Kind
    public var type: Type
    public var initializer: Literal

    public init(name: String, kind: Kind, type: Type, initializer: Literal) {
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
