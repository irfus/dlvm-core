//
//  ResultManager.swift
//  DLVM
//
//  Copyright 2016-2018 The DLVM Team.
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

internal struct PreservedPassResults {
    private typealias ID = ObjectIdentifier

    private var resultMap: [ID : Any] = [:]

    func result<Pass : CacheablePass>(from _: Pass.Type) -> Pass.Result? {
        /// If result is cached, return it
        return resultMap[ID(Pass.self)] as? Pass.Result
    }

    func containsResult<Pass : CacheablePass>(from type: Pass.Type) -> Bool {
        return result(from: type) != nil
    }

    mutating func insert<Pass : CacheablePass>(_ result: Pass.Result,
                                              for type: Pass.Type) {
        resultMap[ID(type)] = result
    }

    mutating func removeResult<Pass : CacheablePass>(from type: Pass.Type) -> Pass.Result? {
        return resultMap.removeValue(forKey: ID(type)) as? Pass.Result
    }

    mutating func removeAll() {
        resultMap.removeAll()
    }
}

public class PassManager<Body> {
    fileprivate var analyses: PreservedPassResults = PreservedPassResults()
    internal init() {}
}

public extension PassManager {
    func invalidateAll() {
        analyses.removeAll()
    }

    @discardableResult
    func invalidateResult<Pass : CacheablePass>(from pass: Pass.Type) -> Pass.Result? {
        return analyses.removeResult(from: pass)
    }

    func updateResult<Pass : CacheablePass>(_ result: Pass.Result, from pass: Pass.Type) {
        analyses.insert(result, for: pass)
    }

    func result<Pass : CacheablePass>(from pass: Pass.Type) -> Pass.Result? {
        return analyses.result(from: pass)
    }
}

public protocol PassResultCache {
    var passManager: PassManager<Self> { get }
    func invalidatePassResults()
}

public extension IRCollection {
    func analysis<P : AnalysisPass>(from _: P.Type) -> P.Result
        where P.Body == Self
    {
        if let result = passManager.result(from: P.self) {
            return result
        }
        let result = P.run(on: self)
        passManager.updateResult(result, from: P.self)
        return result
    }

    @discardableResult
    func runVerification<P : VerificationPass>(_: P.Type) throws -> P.Result
        where P.Body == Self
    {
        if let result = passManager.result(from: P.self) {
            return result
        }
        let result = try P.run(on: self)
        passManager.updateResult(result, from: P.self)
        return result
    }
}

extension PassResultCache {
    func invalidateLocalPassResults() {
        passManager.invalidateAll()
    }
}

extension Module : PassResultCache {
    public func invalidatePassResults() {
        invalidateLocalPassResults()
        for child in self {
            child.invalidateLocalPassResults()
            for grandchild in self {
                grandchild.invalidateLocalPassResults()
            }
        }
    }
}

extension Function : PassResultCache {
    public func invalidatePassResults() {
        invalidateLocalPassResults()
        parent.invalidateLocalPassResults()
        for child in self {
            child.invalidateLocalPassResults()
        }
    }
}

extension BasicBlock : PassResultCache {
    public func invalidatePassResults() {
        invalidateLocalPassResults()
        parent.invalidateLocalPassResults()
        parent.parent.invalidateLocalPassResults()
    }
}
