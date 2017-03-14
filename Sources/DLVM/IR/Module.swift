//
//  Module.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

import Foundation

/// Module representing a neural network
public final class Module : IRCollection, IRUnit {
    public typealias Element = Function
    public typealias Index = Int
    
    open var name: String

    public var elements: OrderedMapSet<Function> = []
    public fileprivate(set) var globals: OrderedMapSet<GlobalValue> = []
    public private(set) var analysisManager: AnalysisManager<Module> = AnalysisManager()
    public internal(set) var transformManager: TransformManager<Module> = TransformManager()

    public init(name: String) {
        self.name = name
    }
}

// MARK: - Globals
extension Module {

    open func insert(_ global: GlobalValue) {
        globals.append(global)
    }

    open func index(of global: GlobalValue) -> Int? {
        return globals.index(of: global)
    }
    
    open func remove(_ global: GlobalValue) {
        globals.remove(global)
    }
    
    open func global(named name: String) -> GlobalValue? {
        return globals.element(named: name)
    }

    open func contains(_ global: GlobalValue) -> Bool {
        return globals.contains(global)
    }

}

// MARK: - Output
extension Module {

    open func write(toFile path: String) throws {
        var contents = ""
        write(to: &contents)
        try contents.write(toFile: path, atomically: true, encoding: .utf8)
    }
    
}
