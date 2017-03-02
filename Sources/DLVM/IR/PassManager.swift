//
//  PassManager.swift
//  DLVM
//
//  Created by Richard Wei on 2/18/17.
//

import Foundation

private typealias ID = ObjectIdentifier

public struct PreservedAnalyses {
    private var analysisMap: [ObjectIdentifier : PassResult] = [:]

    public func analysis<Pass : PassProtocol>(from _: Pass.Type) -> Pass.Result? {
        return analysisMap[ID(Pass.self)] as? Pass.Result
    }

    public func hasAnalysis<Pass : PassProtocol>(from type: Pass.Type) -> Bool {
        return analysis(from: type) != nil
    }

    internal mutating func insert<Pass : PassProtocol>(_ analysis: Pass.Result, for type: Pass.Type) {
        analysisMap[ID(type)] = analysis
    }

    internal mutating func removeAnalysis<Pass : PassProtocol>(for type: Pass.Type) {
        analysisMap.removeValue(forKey: ID(type))
    }

    internal mutating func removeAll() {
        analysisMap.removeAll()
    }
}

open class PassManager {

    private static var instances: [Unowned<Module> : PassManager] = [:]

    private init() {}

    internal static func hasInstance(for module: Module) -> Bool {
        return instances.keys.contains(Unowned(module))
    }

    internal static func removeInstance(for module: Module) {
        instances.removeValue(forKey: Unowned(module))
    }

    open static func `default`(for module: Module) -> PassManager {
        let key = Unowned(module)
        if let pm = instances[key] { return pm }
        let pm = PassManager()
        instances[key] = pm
        return pm
    }

    fileprivate var analysesMap: [Unowned<AnyObject> : PreservedAnalyses] = [:]

}

public extension PassManager {

    func invalidateAllAnalyses() {
        for key in analysesMap.keys {
            analysesMap[key]?.removeAll()
        }
    }

    func removeAllUnits() {
        analysesMap.removeAll()
    }

    func removeUnit<Unit : IRUnit>(_ unit: Unit) {
        analysesMap.removeValue(forKey: Unowned(unit))
    }

    func invalidateAnalyses<Unit : IRUnit>(for unit: Unit) {
        analysesMap[Unowned(unit)]?.removeAll()
    }

    func invalidate<Unit : IRUnit, Pass : PassProtocol>(_ analysis: Pass.Type, for unit: Unit) {
        analysesMap[Unowned(unit)]?.removeAnalysis(for: analysis)
    }

    func updateAnalyses<Unit : IRUnit>(_ analyses: PreservedAnalyses, for unit: Unit) {
        analysesMap[Unowned(unit)] = analyses
    }

    func analyses<Unit : IRUnit>(for unit: Unit) -> PreservedAnalyses {
        let key: Unowned<AnyObject> = Unowned(unit)
        if let analyses = analysesMap[key] { return analyses }
        let analyses = PreservedAnalyses()
        updateAnalyses(analyses, for: unit)
        return analyses
    }

}
