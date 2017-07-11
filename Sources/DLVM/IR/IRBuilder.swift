//
//  IRBuilder.swift
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

import CoreTensor

open class IRBuilder {

    open let module: Module

    /// Current basic block to insert instructions into
    open weak var currentBlock: BasicBlock? {
        didSet {
            insertionIndex = nil
        }
    }

    /// Index of the insertion destination in the current basic block.
    /// When non-nil, this index will be incremented by 1 every time
    /// a 'build' function is invoked.
    open var insertionIndex: Int?

    /// Current function, always the parent of `currentBlock`
    open weak var currentFunction: Function? {
        return currentBlock?.parent
    }

    public init(module: Module) {
        self.module = module
    }

}

public extension IRBuilder {

    convenience init(moduleName: String) {
        self.init(module: Module(name: moduleName))
    }

    convenience init(function: Function) {
        self.init(module: function.parent)
    }

    convenience init(basicBlock: BasicBlock) {
        self.init(module: basicBlock.module)
        move(to: basicBlock)
    }

}

// MARK: - Main builder API
extension IRBuilder {

    @discardableResult
    open func buildStruct(named name: String, fields: DictionaryLiteral<String, Type>) -> StructType {
        let structTy = StructType(name: name, fields: fields.map{$0})
        module.structs.append(structTy)
        return structTy
    }

    @discardableResult
    open func buildAlias(named name: String, for type: Type? = nil) -> Type {
        let alias = TypeAlias(name: name, type: type)
        module.typeAliases.append(alias)
        return .alias(alias)
    }

    @discardableResult
    open func buildGlobalValue(named name: String,
                               type: Type) -> Variable {
        let value = Variable(name: name, type: type)
        module.variables.append(value)
        return value
    }

    @discardableResult
    open func buildFunction(named name: String,
                            argumentTypes: [Type],
                            returnType: Type = .void,
                            attributes: Set<Function.Attribute> = [],
                            declarationKind: Function.DeclarationKind? = nil) -> Function {
        let fun = Function(name: name,
                           argumentTypes: argumentTypes,
                           returnType: returnType,
                           attributes: attributes,
                           declarationKind: declarationKind,
                           parent: module)
        module.append(fun)
        return fun
    }

    @discardableResult
    open func buildBasicBlock(named name: String,
                              arguments: DictionaryLiteral<String, Type>,
                              in function: Function) -> BasicBlock {
        let block = BasicBlock(name: name,
                               arguments: arguments.map{($0.0, $0.1)},
                               parent: function)
        function.append(block)
        return block
    }

    @discardableResult
    open func buildEntry(argumentNames: [String], in function: Function) -> BasicBlock {
        let entry = BasicBlock(name: "entry",
                               arguments: Array(zip(argumentNames, function.argumentTypes)),
                               parent: function)
        function.insert(entry, at: 0)
        return entry
    }

    @discardableResult
    open func buildInstruction(_ kind: InstructionKind, name: String? = nil) -> Instruction {
        guard let block = currentBlock else {
            preconditionFailure("Builder isn't positioned at a basic block")
        }
        let inst = Instruction(name: name, kind: kind, parent: block)
        if let index = insertionIndex {
            block.insert(inst, at: index)
            /// Advance the index
            insertionIndex = index + 1
        } else {
            block.append(inst)
        }
        return inst
    }

}

// MARK: - Positioning
public extension IRBuilder {
    /// Move the builder's insertion point
    func move(to basicBlock: BasicBlock, index: Int? = nil) {
        currentBlock = basicBlock
        insertionIndex = index
    }

    /// Move the builder's insertion point before the specified instruction
    /// - Precondition: The instruction must exist in its parent basic block
    func move(after instruction: Instruction) {
        move(to: instruction.parent, index: instruction.indexInParent + 1)
    }

    /// Move the builder's insertion point
    /// - Precondition: The instruction must exist in its parent basic block
    func move(before instruction: Instruction) {
        move(to: instruction.parent, index: instruction.indexInParent)
    }
}

// MARK: - Op sugar
/// - Note: This extension is only providing limited sugar functions 
/// for common instructions. For full power, please use `buildInstruction`
/// with the algebraic data type `InstructionKind`
public extension IRBuilder {
    func add(_ lhs: Use, _ rhs: Use) -> Instruction {
        return buildInstruction(.zipWith(.associative(.add), lhs, rhs))
    }

    func subtract(_ lhs: Use, _ rhs: Use) -> Instruction {
        return buildInstruction(.zipWith(.associative(.subtract), lhs, rhs))
    }

    func multiply(_ lhs: Use, _ rhs: Use) -> Instruction {
        return buildInstruction(.zipWith(.associative(.multiply), lhs, rhs))
    }

    func divide(_ lhs: Use, _ rhs: Use) -> Instruction {
        return buildInstruction(.zipWith(.associative(.divide), lhs, rhs))
    }

    func power(_ lhs: Use, _ rhs: Use) -> Instruction {
        return buildInstruction(.zipWith(.associative(.power), lhs, rhs))
    }

    func compare(_ operator: ComparisonOp, _ lhs: Use, _ rhs: Use) -> Instruction {
        return buildInstruction(.zipWith(.comparison(`operator`), lhs, rhs))
    }

    func apply(_ function: Use, _ arguments: [Use]) -> Instruction {
        return buildInstruction(.apply(function, arguments))
    }

    func matrixMultiply(_ lhs: Use, _ rhs: Use) -> Instruction {
        return buildInstruction(.matrixMultiply(lhs, rhs))
    }

    func transpose(_ use: Use) -> Instruction {
        return buildInstruction(.transpose(use))
    }

    func exp(_ use: Use) -> Instruction {
        return buildInstruction(.map(.exp, use))
    }

    func map(_ operation: UnaryOp, _ use: Use) -> Instruction {
        return buildInstruction(.map(operation, use))
    }

    func `return`(_ use: Use? = nil) {
        buildInstruction(.return(use))
    }

    func branch(_ destination: BasicBlock, _ arguments: [Use]) {
        buildInstruction(.branch(destination, arguments))
    }

    func conditional(_ condition: Use,
                     then thenBB: BasicBlock, arguments thenArguments: [Use],
                     else elseBB: BasicBlock, arguments elseArguments: [Use]) {
        buildInstruction(.conditional(condition,
                                      thenBB, thenArguments,
                                      elseBB, elseArguments))
    }

    func extract(from source: Use, at indices: [ElementKey]) -> Instruction {
        return buildInstruction(.extract(from: source, at: indices))
    }

    func insert(_ source: Use, to destination: Use, at indices: [ElementKey]) -> Instruction {
        return buildInstruction(.insert(source, to: destination, at: indices))
    }

    func elementPointer(from source: Use, at indices: [ElementKey]) -> Instruction {
        return buildInstruction(.elementPointer(source, indices))
    }

    func load(from source: Use) -> Instruction {
        return buildInstruction(.load(source))
    }

    func store(_ source: Use, to destination: Use) {
        buildInstruction(.store(source, to: destination))
    }

    func bitCast(_ source: Use, to targetType: Type) -> Instruction {
        return buildInstruction(.bitCast(source, targetType))
    }

    func shapeCast(_ source: Use, to targetShape: TensorShape) -> Instruction {
        return buildInstruction(.shapeCast(source, targetShape))
    }

    func dataTypeCast(_ source: Use, to targetDataType: DataType) -> Instruction {
        return buildInstruction(.dataTypeCast(source, targetDataType))
    }

    func allocateHeap(for type: Type, count: Use) -> Instruction {
        return buildInstruction(.allocateHeap(type, count: count))
    }

    func allocateBox(for type: Type) -> Instruction {
        return buildInstruction(.allocateBox(type))
    }

    func projectBox(_ box: Use) -> Instruction {
        return buildInstruction(.projectBox(box))
    }

    func deallocate(_ use: Use) {
        buildInstruction(.deallocate(use))
    }

}
