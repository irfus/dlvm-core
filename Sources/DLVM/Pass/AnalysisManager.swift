//
//  AnalysisManager.swift
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

private typealias ID = ObjectIdentifier

internal struct PreservedAnalyses {
    private var analysisMap: [ObjectIdentifier : Any] = [:]

    func result<Pass : PassProtocol>(from _: Pass.Type) -> Pass.Result? {
        /// If result is cached, return it
        return analysisMap[ID(Pass.self)] as? Pass.Result
    }

    func containsResult<Pass : PassProtocol>(from type: Pass.Type) -> Bool {
        return result(from: type) != nil
    }

    mutating func insert<Pass : PassProtocol>(_ result: Pass.Result,
                                              for type: Pass.Type) {
        analysisMap[ID(type)] = result
    }

    mutating func removeResult<Pass : PassProtocol>(from type: Pass.Type) -> Pass.Result? {
        return analysisMap.removeValue(forKey: ID(type)) as? Pass.Result
    }

    mutating func removeAll() {
        analysisMap.removeAll()
    }
}

public class AnalysisManager<Body : IRUnit> {
    fileprivate var analyses: PreservedAnalyses = PreservedAnalyses()
    internal init() {}
}


public extension AnalysisManager {

    public typealias PassType<Result> = AnalysisPass<Body, Result>

    func invalidateAll() {
        analyses.removeAll()
    }

    @discardableResult
    func invalidateAnalysis<Pass, Result>(from pass: Pass.Type) -> Result?
        where Pass : PassType<Result> {
        return analyses.removeResult(from: pass)
    }

    func updateAnalysis<Pass, Result>(_ result: Result, from pass: Pass.Type)
        where Pass : PassType<Result> {
        analyses.insert(result, for: pass)
    }

    func analysis<Pass, Result>(from pass: Pass.Type) -> Result?
        where Pass : PassType<Result> {
        return analyses.result(from: pass)
    }

}

public extension IRUnit {
    public func analysis<Pass, Result>(from pass: Pass.Type) throws -> Result
        where Pass : AnalysisManager<Self>.PassType<Result> {
        if let result = analysisManager.analysis(from: pass) {
            return result
        }
        let result = try Pass.run(on: self)
        analysisManager.updateAnalysis(result, from: pass)
        return result
    }
}

public extension IRUnit {
    internal func invalidateLocalAnalyses() {
        analysisManager.invalidateAll()
    }
}

public extension IRSubUnit {
    func invalidateAnalyses() {
        parent.invalidateLocalAnalyses()
        analysisManager.invalidateAll()
    }
    
    internal func invalidateLocalAnalyses() {
        parent.invalidateLocalAnalyses()
    }
}

public extension IRUnit where Self : IRCollection, Self.Iterator.Element : IRUnit {
    func invalidateAnalyses() {
        for section in self {
            section.invalidateAnalyses()
        }
    }
}

public extension IRSubUnit where Self : IRCollection, Self.Iterator.Element : IRUnit {
    func invalidateAnalyses() {
        parent.invalidateLocalAnalyses()
        for section in self {
            section.invalidateAnalyses()
        }
    }
}
