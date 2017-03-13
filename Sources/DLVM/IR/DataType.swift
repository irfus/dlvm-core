//
//  DataType.swift
//  DLVM
//
//  Created by Richard Wei on 12/20/16.
//
//

public struct DataType {
    public enum Base : Int { case bool, int, float }
    public var base: Base
    public var size: Int
}

public extension DataType.Base {
    var isNumeric: Bool {
        return self != .bool
    }
}

public extension DataType {
    
    static var bool: DataType {
        return self.init(base: .bool, size: 1)
    }
    
    static func int(_ size: Int) -> DataType {
        return self.init(base: .int, size: size)
    }

    static func float(_ size: Int) -> DataType {
        return self.init(base: .float, size: size)
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
        return lhs.base == rhs.base && lhs.size == rhs.size
    }
}

public extension DataType {
    func canCast(to other: DataType) -> Bool {
        return size <= other.size
            && base.rawValue <= other.base.rawValue
    }
}
