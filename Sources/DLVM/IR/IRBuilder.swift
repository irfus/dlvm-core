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
        _module.updateAnalysisInformation()
        return _module
    }

    open fileprivate(set) weak var currentBlock: BasicBlock? {
        didSet {
            currentFunction = currentBlock?.function
        }
    }

    open weak var currentFunction: Function? {
        didSet {
            if oldValue !== currentFunction {
                variableNameId = 0
            }
        }
    }

    open weak var currentSection: Function.Section? {
        return currentBlock?.parent
    }

    fileprivate var variableNameId = 0

    public init(module: Module) {
        _module = module
    }

    public convenience init(moduleName: String) {
        self.init(module: Module(name: moduleName))
    }

    public convenience init(basicBlock: BasicBlock) {
        self.init(module: basicBlock.module)
        move(to: basicBlock)
    }

}

// MARK: - Helpers
extension IRBuilder {

    func makeVariableName(in function: Function) -> String {
        defer { variableNameId += 1 }
        return disambiguatedName(for: "v\(variableNameId)", in: function)
    }

    func disambiguatedName(for name: String, in function: Function, id: Int = 0) -> String {
        let newName = id == 0 ? name : name + ".\(id)"
        return function.containsInstruction(named: newName)
             ? disambiguatedName(for: name, in: function, id: id + 1)
             : newName
    }

}

// MARK: - Main builder API
extension IRBuilder {

    private func build(_ operation: Operation, name: String?) -> Def<Operation> {
        guard let block = currentBlock else {
            preconditionFailure("Current block doesn't exist")
        }
        let def = Def(name: name ?? makeVariableName(in: block.function), value: operation)
        block.append(.operation(def, parent: block))
        return def
    }

    @discardableResult
    open func declare(_ output: Output, name: String) -> Def<Output> {
        let def = Def<Output>(name: name, value: output)
        let global: Global = .output(def)
        _module.insert(global)
        return def
    }

    @discardableResult
    open func declare(_ placeholder: Placeholder, name: String) -> Def<Placeholder> {
        let def = Def<Placeholder>(name: name, value: placeholder)
        let global: Global = .placeholder(def)
        _module.insert(global)
        return def
    }

    @discardableResult
    open func declare(_ value: GlobalValue, name: String) -> Use {
        let def = Def<GlobalValue>(name: name, value: value)
        _module.insert(.value(def))
        let use = Use(kind: .global(def))
        return use
    }

    open func makeLiteral(_ literal: Literal, shape: TensorShape, type: DataType) -> Use {
        return Use(kind: .literal(LiteralValue(shape: shape, type: type, literal: literal)))
    }

    @discardableResult
    open func buildFunction(named name: String,
                            arguments: [(String, Argument)],
                            result: Argument?) -> Function {
        let fun = Function(name: name, arguments: arguments, result: result, parent: module)
        _module.append(fun)
        return fun
    }

    @discardableResult
    open func buildBasicBlock(named name: String, in function: Function) -> BasicBlock {
        let block = BasicBlock(name: disambiguatedName(for: name, in: function), parent: function)
        function.append(block)
        return block
    }

    @discardableResult
    open func buildBasicBlock(named name: String, in section: Function.Section) -> BasicBlock {
        let block = BasicBlock(name: disambiguatedName(for: name, in: section.parent), parent: section)
        section.append(block)
        return block
    }

    @discardableResult
    open func buildBackwardPass(withRespectTo variable: DifferentiationVariable,
                                in function: Function) -> Function.Section {
        let section = Function.Section(variable: variable, parent: function)
        function.backwardPasses[variable] = section
        return section
    }

    @discardableResult
    open func buildOperation(_ operation: Operation, name: String? = nil) -> Use {
        let def = build(operation, name: name)
        let use = Use(kind: .local(def))
        return use
    }

    open func buildControl(_ control: Control) {
        guard let block = currentBlock else {
            preconditionFailure("Current block doesn't exist")
        }
        currentBlock?.append(.control(control, parent: block))
    }
}

// MARK: - Positioning
extension IRBuilder {

    open func move(to basicBlock: BasicBlock?) {
        currentBlock = basicBlock
    }

}
