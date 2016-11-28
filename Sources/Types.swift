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
    case half   = 2 /// Unimplemented

    init(_ dataType: cudnnDataType_t) {
        self.init(rawValue: dataType.rawValue)!
    }

    var cType: cudnnDataType_t {
        return cudnnDataType_t(rawValue: rawValue)
    }
}

public protocol Randomizable {
    static func random(from lowerBound: Self, to upperBound: Self) -> Self
}

/// Protocol that specifies requirements for the type of elements of the tensor
public protocol TensorDataProtocol
    : KernelDataProtocol, BLASDataProtocol, FloatingPoint, Randomizable,
      ExpressibleByIntegerLiteral, ExpressibleByFloatLiteral {
    static var tensorDataType: TensorDataType { get }
}

import func Foundation.drand48
import func Foundation.srand48
import func Foundation.time

/// Lazy random seed
fileprivate let seed: Void = srand48(time(nil))

extension Float : Randomizable {

    /// Generate random value
    ///
    /// - Parameters:
    ///   - lowerBound: lower bound of the random number
    ///   - upperBound: upper bound (non-inclusive) of the random number
    /// - Returns: random value
    public static func random(from lowerBound: Float, to upperBound: Float) -> Float {
        _ = seed
        return Float(drand48()).truncatingRemainder(dividingBy: upperBound) + lowerBound
    }

}

extension Double : Randomizable {
    
    /// Generate random value
    ///
    /// - Parameters:
    ///   - lowerBound: lower bound of the random number
    ///   - upperBound: upper bound (non-inclusive) of the random number
    /// - Returns: random value
    public static func random(from lowerBound: Double, to upperBound: Double) -> Double {
        _ = seed
        return drand48().truncatingRemainder(dividingBy: upperBound) + lowerBound
    }
    
}

extension Float : TensorDataProtocol {
    
    public static var tensorDataType: TensorDataType {
        return .float
    }
    
}

extension Double : TensorDataProtocol {

    public static var tensorDataType: TensorDataType {
        return .double
    }
    
}
