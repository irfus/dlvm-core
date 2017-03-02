//
//  Pass.swift
//  DLVM
//
//  Created by Richard Wei on 1/15/17.
//
//

import Foundation

public typealias PassResult = Any
public typealias AnyPass = AnyObject

public protocol PassProtocol : class {
    associatedtype Body : IRUnit
    associatedtype Result : PassResult
    static var name: String { get }
    static var dependencies: [AnyPass.Type] { get }
    var identifier: Int? { get set }
    func run(on body: Body) -> Result
    var shouldInvalidateAnalyses: Bool { get }
}

open class Pass<Body : IRUnit, Result : PassResult> : PassProtocol {

    open static var name: String {
        return "Analysis"
    }

    open var identifier: Int?

    open static var dependencies: [AnyPass.Type] {
        return []
    }

    open func run(on body: Body) -> Result {
        fatalError("Unimplemented")
    }

    open var shouldInvalidateAnalyses: Bool {
        return false
    }

}

public typealias AnalysisPass<Body : IRUnit, Result : PassResult> = Pass<Body, Result>

open class TransformationPass<Body : IRUnit> : Pass<Body, Bool> {
    open override var shouldInvalidateAnalyses: Bool {
        return true
    }
}
