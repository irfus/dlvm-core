//
//  Module.swift
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

/// Module representing a neural network
public final class Module : IRCollection {
    public enum Stage {
        case raw, canonical
    }

    public typealias Element = Function
    public typealias Index = Int

    public var name: String
    public internal(set) var stage: Stage
    public var elements: OrderedSet<Function> = []
    public var variables: OrderedSet<Variable> = []
    public var structs: OrderedSet<StructType> = []
    public var typeAliases: OrderedSet<TypeAlias> = []
    public let analysisManager: AnalysisManager<Module> = AnalysisManager()

    public init(name: String, stage: Stage = .raw) {
        self.name = name
        self.stage = stage
    }

    public var canApplyTransforms: Bool {
        return true
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

// MARK: - Name lookup
/// - Note: Name lookup for IR units is implemented as a cached analysis
/// pass, but we don't yet have caching infrastructure for non-IRUnits.
/// Thus name lookup for types and global variables is temporarily implemented
/// as linear search.
public extension Module {
    func variable(named name: String) -> Variable? {
        return variables.first(where: { $0.name == name })
    }

    func `struct`(named name: String) -> StructType? {
        return structs.first(where: { $0.name == name })
    }

    func typeAlias(named name: String) -> TypeAlias? {
        return typeAliases.first(where: { $0.name == name })
    }
}
