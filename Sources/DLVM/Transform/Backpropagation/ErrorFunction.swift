//
//  ErrorFunction.swift
//  DLVM
//
//  Created by Richard Wei on 2/21/17.
//
//

public enum ErrorFunctionBuilderResult: PassResultProtocol {
    case changed(error: Use)
    case unchanged

    public init() {
        self = .unchanged
    }

    public var changed: Bool {
        switch self {
            case .changed: return true
            case .unchanged: return false
        }
    }
}

public protocol ErrorFunctionBuilder : Pass {
    typealias Body = BasicBlock
    typealias Result = ErrorFunctionBuilderResult
    var output: Use { get }
    var referenceOutput: Use { get }
    init(body: BasicBlock, output: Use, referenceOutput: Use)
}

open class MeanSquaredError : Pass {
    open var body: BasicBlock
    open var output: Use
    open var referenceOutput: Use

    public init(body: BasicBlock, output: Use, referenceOutput: Use) {
        self.body = body
        self.output = output
        self.referenceOutput = referenceOutput
    }

    open func run() -> ErrorFunctionBuilderResult {
        var result = Result()
        guard let builder = makeBuilder() else { return result }
        let diff = builder.buildOperation(.binary(.associative(.arithmetic(.subtract)), referenceOutput, output),
                                          name: "difference")
        let squared = builder.buildOperation(.binary(.associative(.arithmetic(.multiply)), diff, diff),
                                             name: "squared")
        let mean = builder.buildOperation(.reduce(.arithmetic(.mean), squared, axis: nil))
        result = .changed(error: mean)
        return result
    }
}


