//
//  ExecutionEngine.swift
//  DLVM
//
//  Created by Richard Wei on 12/1/16.
//
//

import Foundation

public protocol GraphRunner {

    associatedtype DataType : TensorDataProtocol

    init(graph: Graph<DataType>)

    func run()

}

public protocol ExecutionEngine : GraphRunner {

    // TODO
    
}

public class HPVMExecutionEngine<DataType : TensorDataProtocol> : ExecutionEngine {

    public required init(graph: Graph<DataType>) {
        // TODO
    }

    public func run() {
        // TODO
    }
    
}

#if os(macOS) || os(Linux)

public class CUDAExecutionEngine<DataType : TensorDataProtocol> : ExecutionEngine {

    public required init(graph: Graph<DataType>) {
        // TODO
    }

    public func run() {
        // TODO
    }
    
}

#endif

#if os(macOS) || os(iOS)

public class MetalExecutionEngine<DataType : TensorDataProtocol> : ExecutionEngine {

    public required init(graph: Graph<DataType>) {
        // TODO
    }

    public func run() {

    }
    
}

#endif
