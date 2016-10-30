//
//  Types.swift
//  CuDNN
//
//  Created by Richard Wei on 10/30/16.
//
//

import CCuDNN

public enum TensorDataType : UInt32 {
    case float  = 0
    case double = 1
    case half   = 2

    init(_ dataType: cudnnDataType_t) {
        self.init(rawValue: dataType.rawValue)!
    }

    var cType: cudnnDataType_t {
        return cudnnDataType_t(rawValue: rawValue)
    }
}

public protocol TensorDataProtocol {
    static var tensorDataType: TensorDataType { get }
}

extension Double : TensorDataProtocol {
    public static var tensorDataType: TensorDataType {
        return .double
    }
}

extension Float : TensorDataProtocol {
    public static var tensorDataType: TensorDataType {
        return .float
    }
}
