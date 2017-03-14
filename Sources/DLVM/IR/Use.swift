//
// Created by Richard Wei on 2/24/17.
//

public enum Use {
    case argument(Type, Argument)
    case instruction(Type, Instruction)
    case global(Type, GlobalValue)
    case literal(Type, LiteralValue)
}

// MARK: - Equatable
extension Use : Equatable {
    public static func ==(lhs: Use, rhs: Use) -> Bool {
        switch (lhs, rhs) {
        case let (.argument(t1, v1), .argument(t2, v2)):
            return t1 == t2 && v1 === v2
        case let (.instruction(t1, v1), .instruction(t2, v2)):
            return t1 == t2 && v1 === v2
        case let (.global(t1, v1), .global(t2, v2)):
            return t1 == t2 && v1 === v2
        case let (.literal(t1, v1), .literal(t2, v2)):
            return t1 == t2 && v1 == v2
        default:
            return false
        }
    }
}

// MARK: - Value properties
public extension Use {

    var type: Type {
        switch self {
        case .argument(let t, _),
             .instruction(let t, _),
             .global(let t, _),
             .literal(let t, _):
            return t
        }
    }

    var definition: Definition? {
        switch self {
        case .argument(_, let def as Definition),
             .global(_, let def as Definition),
             .instruction(_, let def as Definition):
            return def
        case .literal:
            return nil
        }
    }

    var value: Value {
        switch self {
        case let .argument(_, val): return val
        case let .global(_, val): return val
        case let .literal(_, val): return val
        case let .instruction(_, val): return val
        }
    }

    var name: String? {
        switch self {
        case let .global(_, def): return def.name
        case let .instruction(_, def): return def.name
        case let .argument(_, def): return def.name
        case .literal: return nil
        }
    }

}
