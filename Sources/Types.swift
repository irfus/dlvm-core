//
//  Types.swift
//  CuDNN
//
//  Created by Richard Wei on 10/30/16.
//
//

import CCuDNN
import CuBLAS
import Warp

/// Tensor data type
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

/// Protocol that specifies requirements for the type of elements of the tensor
public protocol TensorDataProtocol : KernelDataProtocol, BLASDataProtocol {
    static var tensorDataType: TensorDataType { get }
    static var zero: Self { get }
}

extension Double : TensorDataProtocol {
    public static var tensorDataType: TensorDataType {
        return .double
    }
    public static var zero: Double {
        return 0.0
    }
}

extension Float : TensorDataProtocol {
    public static var tensorDataType: TensorDataType {
        return .float
    }
    public static var zero: Float {
        return 0.0
    }
}
