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

    var currentBlock: BasicBlock?

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
    open func declare(_ value: GlobalValue) -> GlobalValue {
        _module.insert(value)
        return value
    }

    @discardableResult
    open func makeBasicBlock(named name: String) -> BasicBlock {
        let block = BasicBlock(name: disambiguatedName(for: name))
        _module.insert(block)
        return block
    }

    @discardableResult
    open func makeOperation(_ operation: Operation, name: String? = nil) -> Use {
        return .local(build(operation, name: name))
    }

}

// MARK: - Positioning
extension IRBuilder {

    open func move(to basicBlock: BasicBlock) {
        currentBlock = basicBlock
    }

}
