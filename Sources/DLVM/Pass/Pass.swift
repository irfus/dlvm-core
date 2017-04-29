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

public typealias AnyPass = AnyObject

public protocol PassProtocol : class {
    associatedtype Body : IRUnit
    associatedtype Result
    static var name: String { get }
    static func run(on body: Body) throws -> Result
    static var shouldInvalidateAnalyses: Bool { get }
}

/// Base class for passes
open class Pass<Body : IRUnit, Result> : PassProtocol {
    /// Pass name is the name of the pass type
    public final class var name: String {
        return String(describing: type(of: self))
    }

    /// Runs pass on body
    open class func run(on body: Body) throws -> Result {
        DLUnimplemented()
    }

    /// Determines if the pass will require invalidation of cached
    /// pass result
    public class var shouldInvalidateAnalyses: Bool {
        return true
    }

    /// Initializer is inaccessible anywhere
    private init() {}
}

/// Analysis passes produce analysis information and does not mutate
/// the body
open class AnalysisPass<Body : IRUnit, Result> : Pass<Body, Result> {
    public override static var shouldInvalidateAnalyses: Bool {
        return false
    }
}

/// Transform passes optionally mutate the body and produce a boolean
/// signifying if the body is mutated
open class TransformPass<Body : IRUnit> : Pass<Body, Bool> {
    public override static var shouldInvalidateAnalyses: Bool {
        return true
    }
}
