//
//  Error.swift
//  CuDNN
//
//  Created by Richard Wei on 10/30/16.
//
//

import CCuDNN

public enum DNNError : UInt32, Error {
    case notInitialized   = 1
    case allocationFailed = 2
    case badParameter     = 3
    case internalError    = 4
    case invalidValue     = 5
    case archMismatch     = 6
    case mappingError     = 7
    case executionFailed  = 8
    case notSupported     = 9
    case licenseError     = 10

    init(_ status: cudnnStatus_t) {
        self.init(rawValue: status.rawValue)!
    }
}

func ensureSuccess(_ status: cudnnStatus_t) throws {
    guard status == CUDNN_STATUS_SUCCESS else {
        throw DNNError(status)
    }
}


func forceSuccess(_ status: cudnnStatus_t) {
    guard status == CUDNN_STATUS_SUCCESS else {
        fatalError(String(describing: DNNError(status)))
    }
}

prefix operator !!

@inline(__always)
prefix func !!(status: cudnnStatus_t) {
    forceSuccess(status)
}
