//
//  Module.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

import Foundation

/// Module representing a neural network
open class Module {
    public typealias Element = Function
    
    open var name: String

    public fileprivate(set) var functions = OrderedNamedObjectSet<Function>()
    public fileprivate(set) var globals = OrderedNamedObjectSet<Global>()

    open weak var mainFunction: Function? {
        return function(named: "main")
    }

    public init(name: String) {
        self.name = name
    }
}

// MARK: - Basic block
extension Module {
    
    open func insert(_ function: Function) {
        if let existingBlock = self.function(named: function.name) {
            remove(existingBlock)
        }
        functions.insert(function)
        function.parent = self
    }

    open func index(of function: Function) -> Int? {
        return functions.index(of: function)
    }
    
    open func remove(_ function: Function) {
        functions.remove(function)
        function.parent = nil
    }

    open func function(named name: String) -> Function? {
        return functions.element(named: name)
    }

    open func containsFunction(named name: String) -> Bool {
        return functions.containsValue(named: name)
    }

    open func contains(_ function: Function) -> Bool {
        return functions.contains(function)
    }

}

// MARK: - Globals
extension Module {

    open func insert(_ global: Global) {
        globals.insert(global)
    }

    open func index(of global: Global) -> Int? {
        return globals.index(of: global)
    }
    
    open func remove(_ global: Global) {
        globals.remove(global)
    }
    
    open func global(named name: String) -> Global? {
        return globals.element(named: name)
    }

    open func contains(_ global: Global) -> Bool {
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
