//
//  IRBuilder.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

open class IRBuilder {
    fileprivate let _module: Module

    open var module: Module {
        // _module.updateAnalysisInformation()
        return _module
    }

    weak var currentBlock: BasicBlock?

    weak var currentFunction: Function? {
        return currentBlock?.parent
    }

    fileprivate var variableNameId: Int = 0
    fileprivate var blockNameId: Int = 0
    fileprivate var nameIdTable: [String : Int] = [:]
    
    public init(moduleName: String) {
        _module = Module(name: moduleName)
    }
}

// MARK: - Helpers
extension IRBuilder {
    
    func makeVariableName() -> String {
        defer { variableNameId += 1 }
        return disambiguatedName(for: "v\(variableNameId)")
    }

    func makeBlockName() -> String {
        defer { blockNameId += 1 }
        return disambiguatedName(for: "BB\(blockNameId)")
    }
    
    func disambiguatedName(for name: String) -> String {
        if let id = nameIdTable[name] {
            nameIdTable[name] = id + 1
            return name + ".\(id)"
        }
        nameIdTable[name] = 1
        return name
    }

}

// MARK: - Main builder API
extension IRBuilder {

    private func build(_ operation: Operation, name: String?) -> Def<Operation> {
        guard let block = currentBlock else {
            preconditionFailure("Current block doesn't exist")
        }
        let def = Def(name: name ?? makeVariableName(), value: operation)
        block.append(.operation(def))
        return def
    }

    @discardableResult
    open func declare(_ placeholder: Placeholder, name: String? = nil) -> Use {
        let def = Def<Placeholder>(name: name ?? makeVariableName(), value: placeholder)
        _module.insert(.placeholder(def))
        let use = Use(kind: .placeholder(def))
        return use
    }

    @discardableResult
    open func declare(_ value: GlobalValue, name: String? = nil) -> Use {
        let def = Def<GlobalValue>(name: name ?? makeVariableName(), value: value)
        _module.insert(.value(def))
        let use = Use(kind: .global(def))
        return use
    }

    @discardableResult
    open func makeFunction(named name: String, argument: Def<Argument>, result: Argument) -> Function {
        let fun = Function(name: name, argument: argument, result: result)
        _module.insert(fun)
        return fun
    }

    @discardableResult
    open func makeBasicBlock(named name: String) -> BasicBlock {
        let block = BasicBlock(name: disambiguatedName(for: name))
        currentFunction?.append(block)
        return block
    }

    @discardableResult
    open func makeOperation(_ operation: Operation, name: String? = nil) -> Use {
        let def = build(operation, name: name)
        let use = Use(kind: .local(def))
        return use
    }

    @discardableResult
    open func makeControl(_ control: Control) {
        currentBlock?.append(.control(control))
    }

}

// MARK: - Positioning
extension IRBuilder {

    open func move(to basicBlock: BasicBlock) {
        currentBlock = basicBlock
    }

}
