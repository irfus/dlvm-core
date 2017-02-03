//
//  IRBuilder.swift
//  DLVM
//
//  Created by Richard Wei on 12/18/16.
//
//

open class IRBuilder {
    open let module: Module

    var contextBlocks: [BasicBlock] = []

    open var currentBlock: BasicBlock? {
        return contextBlocks.last
    }
    
    fileprivate var variableNameId: Int = 0
    fileprivate var blockNameId: Int = 0
    fileprivate var nameIdTable: [String : Int] = [:]
    
    public init(moduleName: String) {
        module = Module(name: moduleName)
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

    func pushContextBlock(_ bb: BasicBlock) {
        contextBlocks.append(bb)
    }

    func popContextBlock() -> BasicBlock? {
        return contextBlocks.popLast()
    }

    func clearContextBlocks() {
        contextBlocks.removeAll()
    }
    
}

// MARK: - Main builder API
extension IRBuilder {
    
    private func build<Inst : Instruction>(_ instruction: Inst) -> Inst {
        guard let block = currentBlock else {
            preconditionFailure("Current block doesn't exist")
        }
        block.append(instruction)
        return instruction
    }
    
    @discardableResult
    open func declare<T : GlobalValue>(_ globalValue: T) -> T {
        module.insert(globalValue)
        return globalValue
    }
    
    @discardableResult
    open func declareInput(name: String, type: DataType, shape: TensorShape) -> Input {
        let input = Input(name: name, type: type, shape: shape)
        module.insert(input)
        return input
    }
    
    @discardableResult
    open func declareOutput(name: String, type: DataType, shape: TensorShape) -> Output {
        let output = Output(name: name, type: type, shape: shape)
        module.insert(output)
        return output
    }
    
    @discardableResult
    open func declareParameter(name: String, type: DataType, shape: TensorShape,
                               initializer: Initializer) -> Parameter {
        let parameter = Parameter(name: name, type: type, shape: shape,
                                  initializer: initializer)
        parameter.name = name
        module.insert(parameter)
        return parameter
    }

    @discardableResult
    open func makeGlobalBasicBlock(named name: String) -> BasicBlock {
        let block = BasicBlock(name: disambiguatedName(for: name))
        clearContextBlocks()
        pushContextBlock(block)
        module.insert(block)
        return block
    }

    @discardableResult
    open func makeExtension(ofType type: BasicBlock.ExtensionType, for basicBlock: BasicBlock) -> BasicBlock {
        return basicBlock.makeExtension(ofType: type)
    }
    
    @discardableResult
    open func makeArithmeticOperation(_ `operator`: ArithmeticOperator,
                                      _ lhs: Value, _ rhs: Value,
                                      name: String? = nil) -> ArithmeticInstruction {
        let inst = ArithmeticInstruction(name: name ?? makeVariableName(),
                                         function: `operator`,
                                         firstOperand: lhs, secondOperand: rhs)
        return build(inst)
    }

    @discardableResult
    open func makeLogicOperation(_ `operator`: LogicOperator,
                                 _ lhs: Value, _ rhs: Value,
                                 name: String? = nil) -> LogicInstruction {
        let inst = LogicInstruction(name: name ?? makeVariableName(),
                                    function: `operator`,
                                    firstOperand: lhs, secondOperand: rhs)
        return build(inst)
    }
    
    @discardableResult
    open func makeComparison(_ `operator`: ComparisonPredicate,
                             _ lhs: Value, _ rhs: Value,
                             name: String? = nil) -> ComparisonInstruction {
        let inst = ComparisonInstruction(name: name ?? makeVariableName(),
                                         function: `operator`,
                                         firstOperand: lhs, secondOperand: rhs)
        return build(inst)
    }
    
    @discardableResult
    open func makeTensorMultiplication(_ lhs: Value, _ rhs: Value,
                                       name: String? = nil) -> TensorMultiplicationInstruction {
        let inst = TensorMultiplicationInstruction(name: name ?? makeVariableName(),
                                                   firstOperand: lhs, secondOperand: rhs)
        return build(inst)
    }
    
    @discardableResult
    open func makeMatrixMultiplication(_ lhs: Value, _ rhs: Value,
                                       name: String? = nil) -> MatrixMultiplicationInstruction {
        let inst = MatrixMultiplicationInstruction(name: name ?? makeVariableName(),
                                                   firstOperand: lhs, secondOperand: rhs)
        return build(inst)
    }
    
    @discardableResult
    open func makeElementwiseTransformation(_ function: ElementwiseFunction,
                                            _ operand: Value,
                                            name: String? = nil) -> ElementwiseInstruction {
        let inst = ElementwiseInstruction(name: name ?? makeVariableName(),
                                          function: function, operand: operand)
        return build(inst)
    }
    
    @discardableResult
    open func makeAggregation(_ function: AggregationFunction,
                              _ operand: Value,
                              name: String? = nil) -> AggregationInstruction {
        let inst = AggregationInstruction(name: name ?? makeVariableName(),
                                          function: function, operand: operand)
        return build(inst)
    }

    @discardableResult
    open func makeReduction(_ function: ReductionFunction, _ operand: Value,
                            name: String? = nil) -> ReductionInstruction {
        let inst = ReductionInstruction(name: name ?? makeVariableName(), function: function, operand: operand)
        return build(inst)
    }

    @discardableResult
    open func makeBinaryReduction(_ function: BinaryIntegrationFunction,
                                  _ firstOperand: Value, _ secondOperand: Value,
                                  name: String? = nil) -> BinaryReductionInstruction {
        let inst = BinaryReductionInstruction(name: name ?? makeVariableName(), function: function,
                                              firstOperand: firstOperand, secondOperand: secondOperand)
        return build(inst)
    }
    
    @discardableResult
    open func makeConcatenation(_ operands: [Value], axis: Int,
                                name: String? = nil) -> ConcatenationInstruction {
        let inst = ConcatenationInstruction(name: name ?? makeVariableName(),
                                            operands: operands, axis: axis)
        return build(inst)
    }
    
    @discardableResult
    open func makeShapeCast(_ operand: Value, targetShape: TensorShape,
                            name: String? = nil) -> ShapeCastInstruction {
        let inst = ShapeCastInstruction(name: name ?? makeVariableName(),
                                        operand: operand, target: targetShape)
        return build(inst)
    }
    
    @discardableResult
    open func makeTypeCast(_ operand: Value, targetType: DataType,
                           name: String? = nil) -> TypeCastInstruction {
        let inst = TypeCastInstruction(name: name ?? makeVariableName(), operand: operand,
                                       target: targetType)
        return build(inst)
    }
    
    @discardableResult
    open func makeLoad(_ source: Input, name: String? = nil) -> LoadInstruction {
        let inst = LoadInstruction(name: name ?? makeVariableName(), source: source)
        return build(inst)
    }
    
    @discardableResult
    open func makeStore(_ source: Value, to destination: Value) -> StoreInstruction {
        let inst = StoreInstruction(source: source, destination: destination)
        return build(inst)
    }

    @discardableResult
    open func beginLoop(named name: String? = nil) -> BasicBlock {
        let bb = BasicBlock(name: name ?? makeBlockName())
        pushContextBlock(bb)
        return bb
    }

    @discardableResult
    open func exitLoop(onCondition condition: LoopInstruction.Condition) -> LoopInstruction {
        guard let bb = popContextBlock() else {
            preconditionFailure("Not in a loop")
        }
        let inst = LoopInstruction(condition: condition, body: bb)
        return build(inst)
    }
    
}
