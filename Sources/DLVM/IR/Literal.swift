//
//  Literal.swift
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

/// Scalar or tensor literal, literally
/// - Note: It has no type or shape, because a `Literal` is not a `Value`.
/// But `LiteralValue`, that uses `Literal`, is a value.
public indirect enum Literal {
    public enum Scalar {
        case int(IntegerLiteralType)
        case float(FloatLiteralType)
        case bool(BooleanLiteralType)
    }
    case undefined
    case null
    case zero
    case scalar(Scalar)
    case tensor([Use])
    case tuple([Use])
    case array([Use])
    case `struct`([(String, Use)])
}

extension Literal : ExpressibleByIntegerLiteral {
    public init(integerLiteral value: IntegerLiteralType) {
        self = .scalar(.int(value))
    }
}

extension Literal : ExpressibleByBooleanLiteral {
    public init(booleanLiteral value: BooleanLiteralType) {
        self = .scalar(.bool(value))
    }
}

extension Literal : ExpressibleByFloatLiteral {
    public init(floatLiteral value: FloatLiteralType) {
        self = .scalar(.float(value))
    }
}

extension Literal : ExpressibleByArrayLiteral {
    public init(arrayLiteral elements: Use...) {
        self = .array(elements)
    }
}

extension Literal : ExpressibleByDictionaryLiteral {
    public init(dictionaryLiteral elements: (String, Use)...) {
        self = .struct(elements)
    }
}

extension Literal : ExpressibleByNilLiteral {
    public init(nilLiteral: ()) {
        self = .null
    }
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
        case let (.array(tt1), .array(tt2)):
            return tt1 == tt2
        case let (.struct(ss1), .struct(ss2)):
            return ss1.elementsEqual(ss2, by: { $0 == $1 })
        default: return false
        }
    }
}

// MARK: - Scalar literal type base
public extension Literal.Scalar {
    var typeBase: DataType.Base {
        switch self {
        case .bool: return .bool
        case .int: return .int
        case .float: return .float
        }
    }
}

// MARK: - Equatable
extension Literal.Scalar : Equatable {
    public static func == (lhs: Literal.Scalar, rhs: Literal.Scalar) -> Bool {
        switch (lhs, rhs) {
        case let (.int(i1), .int(i2)): return i1 == i2
        case let (.float(f1), .float(f2)): return f1 == f2
        case let (.bool(b1), .bool(b2)): return b1 == b2
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
        return .literal(type, literal)
    }
}

extension LiteralValue : Equatable {
    public static func == (lhs: LiteralValue, rhs: LiteralValue) -> Bool {
        return lhs.type == rhs.type
            && lhs.literal == rhs.literal
    }
}

public extension Use {
    static func tensor(_ shape: TensorShape, _ dataType: DataType, repeating item: Int) -> Use {
        let scalLit: Literal.Scalar
        switch dataType.base {
        case .int: scalLit = .int(item)
        case .float: scalLit = .float(Double(item))
        case .bool: scalLit = .bool(item == 0 ? false : true)
        }
        let lit: Literal = .scalar(scalLit)
        let type: Type = .tensor(shape, dataType)

        if shape.isScalar {
            return .literal(type, lit)
        } else {
            let subtensor = Use.tensor(shape.dropFirst(), dataType, repeating: item)
            let subtensors = Array(repeating: subtensor, count: shape[0])
            return .literal(type, .tensor(subtensors))
        }
    }
}

public extension Literal {
    func substituting(_ new: Use, for old: Use) -> Literal {
        let condSubst = {$0 == old ? new : $0}
        switch self {
        case .array(let vv): return .array(vv.map(condSubst))
        case .tensor(let vv): return .array(vv.map(condSubst))
        case .tuple(let vv): return .array(vv.map(condSubst))
        case .struct(let fields):
            return .struct(Array(fields.map{($0.0, condSubst($0.1))}))
        case .null, .undefined, .zero, .scalar: return self
        }
    }
}
