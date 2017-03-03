//
//  AnalysisManager.swift
//  DLVM
//
//  Created by Richard Wei on 2/18/17.
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

open class AnalysisManager<Body : IRUnit> {
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

    public func analysis<Pass, Result>(from pass: Pass.Type) -> Result
        where Pass : AnalysisManager<Self>.PassType<Result> {
        if let result = analysisManager.analysis(from: pass) {
            return result
        }
        let result = Pass.run(on: self)
        analysisManager.updateAnalysis(result, from: pass)
        return result
    }

}

public extension IRUnit {
    func invalidateAnalyses() {
        analysisManager.invalidateAll()
    }
}

public extension IRUnit where Self : IRCollection, Self.Iterator.Element : IRSubUnit {
    func invalidateAnalyses() {
        analysisManager.invalidateAll()
        for element in self {
            element.invalidateAnalyses()
        }
    }
}
