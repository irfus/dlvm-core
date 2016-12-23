//
//  Types.swift
//  DLVM
//
//  Created by Richard Wei on 12/20/16.
//
//

public enum DataType {
    case int1, int8, int16, int32, int64
    case float8, float16, float32, float64
}

public protocol DLVMDataProtocol {
    static var dataType: DataType { get }
}
