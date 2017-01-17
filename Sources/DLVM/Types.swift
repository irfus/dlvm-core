//
//  Types.swift
//  DLVM
//
//  Created by Richard Wei on 12/20/16.
//
//

public enum TypeBase {
    case bool, int, float
}

public protocol DataType : TextOutputStreamable {
    var base: TypeBase { get set }
    var size: Int { get set }
}

public struct ScalarType : DataType, Equatable {
    public var base: TypeBase
    public var size: Int

    public static func ~=(lhs: ScalarType, rhs: ScalarType) -> Bool {
        return lhs.base == rhs.base
    }
    
    public static func ==(lhs: ScalarType, rhs: ScalarType) -> Bool {
        return lhs.base == rhs.base && lhs.size == rhs.size
    }

    public func makeTensorType(with shape: TensorShape) -> TensorType {
        return TensorType(base: base, size: size, shape: shape)
    }
}

extension ScalarType {
    public static var bool: ScalarType {
        return self.init(base: .bool, size: 1)
    }
    
    public static func int(_ size: Int) -> ScalarType {
        return self.init(base: .int, size: size)
    }

    public static func float(_ size: Int) -> ScalarType {
        return self.init(base: .float, size: size)
    }
}

extension DataType {
    public var scalarType: ScalarType {
        return ScalarType(base: base, size: size)
    }
}

public struct TensorType : DataType, Equatable {
    public var base: TypeBase
    public var size: Int
    public var shape: TensorShape

    public static func ==(lhs: TensorType, rhs: TensorType) -> Bool {
        return lhs.base == rhs.base && lhs.size == rhs.size
    }
}
