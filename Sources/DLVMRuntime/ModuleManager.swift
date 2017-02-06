//
//  ModuleManager.swift
//  DLVM
//
//  Created by Richard Wei on 2/3/17.
//
//

import DLVM

open class ModuleManager {

    fileprivate var activeModules: NamedObjectSet<Module> = []

    public init() {
    }

    public func load(_ module: Module) {
        activeModules.insert(module)
    }

    public func unload(_ module: Module) {
        activeModules.remove(module)
    }

}
