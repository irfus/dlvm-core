//
//  Pass.swift
//  DLVM
//
//  Copyright 2016-2017 Richard Wei.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

import Foundation

public typealias AnyPass = AnyObject

public protocol PassProtocol : class {
    associatedtype Body : IRUnit
    associatedtype Result
    static var name: String { get }
    static func run(on body: Body) throws -> Result
    static var shouldInvalidateAnalyses: Bool { get }
}

open class Pass<Body : IRUnit, Result> : PassProtocol {

    open class var name: String {
        return String(describing: type(of: self))
    }

    open class func run(on body: Body) throws -> Result {
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

open class TransformPass<Body : IRUnit> : Pass<Body, Bool> {
    open override class var shouldInvalidateAnalyses: Bool {
        return true
    }
}
