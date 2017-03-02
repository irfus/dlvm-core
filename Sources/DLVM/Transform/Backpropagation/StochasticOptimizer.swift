//
//  StochasticOptimizer.swift
//  DLVM
//
//  Created by Richard Wei on 2/21/17.
//
//

/*

public protocol StochasticOptimizer : Pass {
    typealias Body = BasicBlock
    var gradient: Use { get }
    var target: Def<GlobalValue> { get }
    var learningRate: Double { get }
    init(body: BasicBlock, gradient: Use, target: Def<GlobalValue>, learningRate: Double)
}

public class StochasticGradientDescent : StochasticOptimizer {
    public typealias Body = BasicBlock
    public let gradient: Use
    public let target: Def<GlobalValue>
    public let learningRate: Double
    public let body: BasicBlock

    public required init(body: BasicBlock, gradient: Use, target: Def<GlobalValue>, learningRate: Double) {
        self.body = body
        self.gradient = gradient
        self.target = target
        self.learningRate = learningRate
    }

    public func run() -> PassResult {
        var result = PassResult()
        guard let builder = makeBuilder() else { return result }
        builder.move(to: body)
        let eta = builder.makeLiteral(.scalar(.float(learningRate)),
                                      shape: .scalar, type: .float(32))
        let error = builder.buildOperation(.binary(.associative(.arithmetic(.multiply)), eta, gradient),
                                           name: "err")
        let newVal = builder.buildOperation(.binary(.associative(.arithmetic(.subtract)),
                                            Use(kind: .global(target)), error), name: "new_val")
        builder.buildControl(.store(newVal, to: target))
        result.changed = true
        return result
    }
}

 */
