//
//  IRBuilder.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

open class IRBuilder {

    open let module: Module

    open fileprivate(set) weak var currentBlock: BasicBlock? {
        didSet {
            currentFunction = currentBlock?.parent
        }
    }

    open weak var currentFunction: Function? {
        didSet {
            if oldValue !== currentFunction {
                variableNameId = 0
            }
        }
    }

    fileprivate var variableNameId = 0

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

// MARK: - Helpers
extension IRBuilder {

    func makeVariableName(in function: Function) -> String {
        defer { variableNameId += 1 }
        return disambiguatedName(for: "v\(variableNameId)", in: function)
    }

    func disambiguatedName(for name: String, in function: Function, id: Int = 0) -> String {
        let newName = id == 0 ? name : name + ".\(id)"
        return function.containsName(newName)
             ? disambiguatedName(for: name, in: function, id: id + 1)
             : newName
    }

}

// MARK: - Main builder API
extension IRBuilder {

    @discardableResult
    open func buildGlobalValue(_ value: GlobalValue) -> Use {
        module.globalValues.append(value)
        return .global(value.type, value)
    }

    @discardableResult
    open func buildTypeAlias(_ alias: TypeAlias) -> Type {
        module.typeAliases.append(alias)
        return .alias(alias)
    }

    open func makeLiteral(_ literalValue: LiteralValue) -> Use {
        return .literal(literalValue.type, literalValue)
    }

    open func makeUse(_ argument: Argument) -> Use {
        return .argument(argument.type, argument)
    }

    @discardableResult
    open func buildFunction(named name: String,
                            arguments: [(String, Type)],
                            result: Type = .void,
                            attributes: Set<Function.Attribute>) -> Function {
        let fun = Function(name: name,
                           arguments: arguments,
                           result: result,
                           attributes: attributes,
                           parent: module)
        module.append(fun)
        return fun
    }

    @discardableResult
    open func buildBasicBlock(named name: String,
                              arguments: [(String, Type)],
                              in function: Function) -> BasicBlock {
        if name == "entry" { return function.entry }
        let newName = disambiguatedName(for: name, in: function)
        let block = BasicBlock(name: newName, arguments: arguments, parent: function)
        function.append(block)
        return block
    }

    @discardableResult
    open func buildInstruction(_ kind: InstructionKind, name: String? = nil) -> Use {
        guard let block = currentBlock else {
            preconditionFailure("Builder isn't positioned at a basic block")
        }
        let function = block.parent
        let inst = Instruction(name: kind.type.isVoid ? nil : (name ?? makeVariableName(in: function)),
                               kind: kind, parent: block)
        block.append(inst)
        return .instruction(inst.type, inst)
    }
}

// MARK: - Positioning
extension IRBuilder {
    open func move(to basicBlock: BasicBlock?) {
        currentBlock = basicBlock
    }
}

// MARK: - Op sugar
/// - Note: This extension is only providing limited sugar functions 
/// for common instructions. For full power, please use `buildInstruction`
/// with the algebraic data type `InstructionKind`
public extension IRBuilder {
    func add(_ lhs: Use, _ rhs: Use) -> Use {
        return buildInstruction(.binary(.associative(.arithmetic(.add)), lhs, rhs))
    }

    func subtract(_ lhs: Use, _ rhs: Use) -> Use {
        return buildInstruction(.binary(.associative(.arithmetic(.subtract)), lhs, rhs))
    }

    func multiply(_ lhs: Use, by rhs: Use) -> Use {
        return buildInstruction(.binary(.associative(.arithmetic(.multiply)), lhs, rhs))
    }

    func divide(_ lhs: Use, by rhs: Use) -> Use {
        return buildInstruction(.binary(.associative(.arithmetic(.divide)), lhs, rhs))
    }

    func power(_ lhs: Use, _ rhs: Use) -> Use {
        return buildInstruction(.binary(.associative(.arithmetic(.power)), lhs, rhs))
    }

    func call(_ function: Use, _ arguments: [Use]) -> Use {
        return buildInstruction(.call(function, arguments))
    }

    func gradient(_ function: Use, _ arguments: [Use]) -> Use {
        return buildInstruction(.gradient(function, arguments))
    }

    func matrixMultiply(_ lhs: Use, _ rhs: Use) -> Use {
        return buildInstruction(.matrixMultiply(lhs, rhs))
    }

    func transpose(_ use: Use) -> Use {
        return buildInstruction(.transpose(use))
    }

    func transform(_ operation: ElementwiseOp, _ use: Use) -> Use {
        return buildInstruction(.unary(.elementwise(operation), use))
    }

    func integrate(_ operation: IntegrationOp, _ use: Use) -> Use {
        return buildInstruction(.unary(.integration(operation), use))
    }

    func `return`(_ use: Use? = nil) {
        buildInstruction(.return(use))
    }

    func branch(_ destination: BasicBlock, _ arguments: [Use] = []) {
        buildInstruction(.branch(destination, arguments))
    }

    func conditional(_ condition: Use,
                     then thenBB: BasicBlock, arguments thenArguments: [Use] = [],
                     else elseBB: BasicBlock, arguments elseArguments: [Use] = []) {
        buildInstruction(.conditional(condition,
                                      thenBB, thenArguments,
                                      elseBB, elseArguments))
    }

    func extract(from source: Use, at indices: [Int]) -> Use {
        return buildInstruction(.extract(from: source, at: indices))
    }

    func insert(_ source: Use, to destination: Use, at indices: [Int]) -> Use {
        return buildInstruction(.insert(source, to: destination, at: indices))
    }

    func elementPointer(from source: Use, at indices: [Use]) -> Use {
        return buildInstruction(.elementPointer(source, indices))
    }

    func load(from source: Use) -> Use {
        return buildInstruction(.load(source))
    }

    func store(_ source: Use, to destination: Use) {
        buildInstruction(.store(source, to: destination))
    }

    func bitCast(_ source: Use, to targetType: Type) -> Use {
        return buildInstruction(.bitCast(source, targetType))
    }

    func shapeCast(_ source: Use, to targetShape: TensorShape) -> Use {
        return buildInstruction(.shapeCast(source, targetShape))
    }

    func dataTypeCast(_ source: Use, to targetDataType: DataType) -> Use {
        return buildInstruction(.dataTypeCast(source, targetDataType))
    }

}
