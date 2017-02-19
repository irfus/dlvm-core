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

    open fileprivate(set) weak var currentBlock: BasicBlock?

    open weak var currentFunction: Function? {
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
    open func declare(_ output: Output, name: String? = nil) -> Def<Output> {
        let def = Def<Output>(name: name ?? makeVariableName(), value: output)
        let global: Global = .output(def)
        _module.insert(global)
        return def
    }

    @discardableResult
    open func declare(_ placeholder: Placeholder, name: String? = nil) -> Def<Placeholder> {
        let def = Def<Placeholder>(name: name ?? makeVariableName(), value: placeholder)
        let global: Global = .placeholder(def)
        _module.insert(global)
        return def
    }

    @discardableResult
    open func declare(_ value: GlobalValue, name: String? = nil) -> Use {
        let def = Def<GlobalValue>(name: name ?? makeVariableName(), value: value)
        _module.insert(.value(def))
        let use = Use(kind: .global(def))
        return use
    }

    @discardableResult
    open func buildFunction(named name: String, 
                            arguments: [(String, Argument)], result: Argument?) -> Function {
        let fun = Function(name: name, arguments: arguments, result: result)
        _module.append(fun)
        return fun
    }
    
    @discardableResult
    open func buildBasicBlock(named name: String) -> BasicBlock {
        let block = BasicBlock(name: disambiguatedName(for: name))
        if let currentBlock = currentBlock, let function = currentBlock.parent {
            function.insert(block, after: currentBlock)
        }
        return block
    }

    @discardableResult
    open func buildBasicBlock(named name: String, in function: Function) -> BasicBlock {
        let block = BasicBlock(name: disambiguatedName(for: name))
        function.append(block)
        return block
    }

    @discardableResult
    open func buildOperation(_ operation: Operation, name: String? = nil) -> Use {
        let def = build(operation, name: name)
        let use = Use(kind: .local(def))
        return use
    }

    open func buildControl(_ control: Control) {
        currentBlock?.append(.control(control))
    }

}

// MARK: - Positioning
extension IRBuilder {

    open func move(to basicBlock: BasicBlock?) {
        currentBlock = basicBlock
    }

}
