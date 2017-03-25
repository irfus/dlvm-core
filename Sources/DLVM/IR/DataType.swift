//
//  DataType.swift
//  DLVM
//
//  Created by Richard Wei on 12/20/16.
//
//

public enum FloatingPointSize : Int {
    case half = 16
    case single = 32
    case double = 64
}

extension FloatingPointSize : Comparable {
    public static func <(lhs: FloatingPointSize, rhs: FloatingPointSize) -> Bool {
        return lhs.rawValue < rhs.rawValue
    }
}

public enum DataType {
    public enum Base : Int { case bool, int, float }
    case bool
    case int(Int)
    case float(FloatingPointSize)
}
public extension DataType.Base {
    var isNumeric: Bool {
        return self != .bool
    }
}

public extension DataType {
    var base: Base {
        switch self {
        case .bool: return .bool
        case .int: return .int
        case .float: return .float
        }
    }

    static func ~(lhs: DataType, rhs: DataType) -> Bool {
        return lhs.base == rhs.base
    }

    var isNumeric: Bool {
        return base.isNumeric
    }

    var isBool: Bool {
        return base == .bool
    }

}

extension DataType : Equatable {
    public static func ==(lhs: DataType, rhs: DataType) -> Bool {
        switch (lhs, rhs) {
        case (.bool, .bool): return true
        case let (.int(w1), .int(w2)): return w1 == w2
        case let (.float(w1), .float(w2)): return w1 == w2
        default: return false
        }
    }
}

public extension DataType {
    func canCast(to other: DataType) -> Bool {
        switch (self, other) {
        case (.bool, .bool): return true
        case let (.int(w1), .int(w2)): return w1 <= w2
        case let (.float(w1), .float(w2)): return w1 <= w2
        default: return false
        }
    }
}
