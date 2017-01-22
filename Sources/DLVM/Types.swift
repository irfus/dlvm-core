//
//  Types.swift
//  DLVM
//
//  Created by Richard Wei on 12/20/16.
//
//

public struct DataType : Equatable {
    public enum Base {
        case bool, int, float
    }

    public var base: Base
    public var size: Int

    public init(base: Base, size: Int) {
        self.base = base
        self.size = size
    }
    
    public static var bool: DataType {
        return self.init(base: .bool, size: 1)
    }
    
    public static func int(_ size: Int) -> DataType {
        return self.init(base: .int, size: size)
    }

    public static func float(_ size: Int) -> DataType {
        return self.init(base: .float, size: size)
    }

    public static func ~=(lhs: DataType, rhs: DataType) -> Bool {
        return lhs.base == rhs.base
    }

    public static func ==(lhs: DataType, rhs: DataType) -> Bool {
        return lhs.base == rhs.base && lhs.size == rhs.size
    }

}
