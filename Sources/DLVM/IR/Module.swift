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
    public var globalValues: OrderedMapSet<GlobalValue> = []
    public var typeAliases: OrderedMapSet<TypeAlias> = []
    public private(set) var analysisManager: AnalysisManager<Module> = AnalysisManager()
    public internal(set) var transformManager: TransformManager<Module> = TransformManager()

    public init(name: String) {
        self.name = name
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
