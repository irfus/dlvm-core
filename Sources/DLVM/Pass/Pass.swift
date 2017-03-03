//
//  Pass.swift
//  DLVM
//
//  Created by Richard Wei on 1/15/17.
//
//

import Foundation

public typealias AnyPass = AnyObject

public protocol PassProtocol : class {
    associatedtype Body : IRUnit
    associatedtype Result
    static var name: String { get }
    static func run(on body: Body) -> Result
    static var shouldInvalidateAnalyses: Bool { get }
}

open class Pass<Body : IRUnit, Result> : PassProtocol {

    open class var name: String {
        return String(describing: type(of: self))
    }

    open class func run(on body: Body) -> Result {
        fatalError("Unimplemented")
    }

    open class var shouldInvalidateAnalyses: Bool {
        return false
    }

    private init() {}

}

open class AnalysisPass<Body : IRUnit, Result> : Pass<Body, Result> {
    public final var shouldInvalidateAnalyses: Bool {
        return false
    }
}

open class TransformationPass<Body : IRUnit> : Pass<Body, Bool> {
    open override class var shouldInvalidateAnalyses: Bool {
        return true
    }
}
