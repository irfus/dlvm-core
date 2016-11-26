//
//  DNN.swift
//  LLNM
//
//  Created by Richard Wei on 11/6/16.
//
//

import CCuDNN
import CUDARuntime
import protocol CUDADriver.CHandleCarrier

/// cuDNN Wrapper
final class DNN : CHandleCarrier {

    private static var instances: [DNN?] = Array(repeating: nil, count: Device.count)

    class func shared(on device: Device) -> DNN {
        if let dnn = instances[device.index] {
            return dnn
        }
        let dnn = DNN(device: device)
        instances[device.index] = dnn
        return dnn
    }

    let handle: cudnnHandle_t
    let device: Device

    private init() {
        var handle: Handle?
        !!cudnnCreate(&handle)
        self.handle = handle!
        self.device = Device.current
    }

    private convenience init(device: Device) {
        Device.current = device
        self.init()
    }

    deinit {
        cudnnDestroy(handle)
    }

    func withUnsafeHandle<Result>(_ body: (cudnnHandle_t) throws -> Result) rethrows -> Result {
        return try body(handle)
    }
    
}
