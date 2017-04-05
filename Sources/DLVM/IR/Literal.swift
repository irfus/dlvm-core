//
//  Literal.swift
//  DLVM
//
//  Created by Richard Wei on 3/16/17.
//
//

/// Scalar literal
public enum ScalarLiteral {
    case int(IntegerLiteralType)
    case float(FloatLiteralType)
    case bool(BooleanLiteralType)

    public var typeBase: DataType.Base {
        switch self {
        case .bool: return .bool
        case .int: return .int
        case .float: return .float
        }
    }
}

extension ScalarLiteral : Equatable {
    public static func == (lhs: ScalarLiteral, rhs: ScalarLiteral) -> Bool {
        switch (lhs, rhs) {
        case let (.int(i1), .int(i2)): return i1 == i2
        case let (.float(f1), .float(f2)): return f1 == f2
        case let (.bool(b1), .bool(b2)): return b1 == b2
        default: return false
        }
    }
}

/// Scalar or tensor literal, literally
/// - Note: It has no type or shape, because a `Literal` is not a `Value`.
/// But `LiteralValue`, that uses `Literal`, is a value.
public indirect enum Literal {
    case undefined
    case null
    case zero
    case scalar(ScalarLiteral)
    case tensor([Use])
    case tuple([Use])
    case `struct`([Use], isPacked: Bool)
    case array([Use])
    case constant(InstructionKind)
}

extension Literal : Equatable {
    public static func == (lhs: Literal, rhs: Literal) -> Bool {
        switch (lhs, rhs) {
        case (.zero, .zero),
             (.undefined, .undefined),
             (.null, .null):
            return true
        case let (.scalar(s1), .scalar(s2)):
            return s1 == s2
        case let (.tensor(t1), .tensor(t2)):
            return t1 == t2
        case let (.tuple(tt1), .tuple(tt2)):
            return tt1 == tt2
        case let (.struct(vv1, p1), .struct(vv2, p2)):
            return vv1 == vv2 && p1 == p2
        case let (.array(tt1), .array(tt2)):
            return tt1 == tt2
        case (.constant(_), .constant(_)):
            return false /// Or rather unimplemented
        default: return false
        }
    }
}

/// Literal value. It wraps `Literal` into a value
public struct LiteralValue : Value {
    public var type: Type
    public var literal: Literal

    public init(type: Type, literal: Literal) {
        self.type = type
        self.literal = literal
    }

    public func makeUse() -> Use {
        return .literal(type, self)
    }
}

extension LiteralValue : Equatable {
    public static func == (lhs: LiteralValue, rhs: LiteralValue) -> Bool {
        return lhs.type == rhs.type
            && lhs.literal == rhs.literal
    }
}

public extension LiteralValue {
    init(shape: TensorShape, dataType: DataType, repeating number: Int) {
        let scalLit: ScalarLiteral
        switch dataType.base {
        case .int: scalLit = .int(number)
        case .float: scalLit = .float(Double(number))
        case .bool: scalLit = .bool(number == 0 ? false : true)
        }
        let lit: Literal = .scalar(scalLit)
        let type: Type = .tensor(shape, dataType)

        if shape.isScalar {
            self.init(type: type, literal: lit)
        } else {
            let subtensor = LiteralValue(shape: shape.dropFirst(),
                                         dataType: dataType,
                                         repeating: number)
            let subtensors = Array(repeating: subtensor.makeUse(), count: shape[0])
            self.init(type: type, literal: .tensor(subtensors))
        }
    }
}
