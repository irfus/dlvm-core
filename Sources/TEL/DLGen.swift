//
//  DLGen.swift
//  DLVM
//
//  Created by Richard Wei on 12/23/16.
//
//  This file contains DLVM IR generator from TEL
//

import DLVM

public extension Program {
    public func makeModule() -> Module {
        let cgen = CodeGenerator(program: self)
        let module = cgen.makeModule()
        return module
    }
}

struct CodeGenEnvironment {
    var variables: [String : VariableOperand] = [:]
    
    subscript(key: String) -> VariableOperand? {
        get {
            return variables[key]
        }
        set {
            variables[key] = newValue
        }
    }
}

class CodeGenerator {
    
    let program: Program
    lazy var builder: IRBuilder = IRBuilder(moduleName: self.program.moduleName)
    var environment = CodeGenEnvironment()

    init(program: Program) {
        self.program = program
    }
    
    func makeModule() -> Module {
        /// Declare input
        for input in program.inputs {
            let variable = builder.declareTensor(named: input.name,
                                                 dataType: program.dataType,
                                                 shape: input.shape)
            environment[variable.name] = variable
        }
        /// Define globals
        for param in program.parameters {
            let def: TensorDefinition
            switch param.initializer {
            case let .constant(.int(i)):
                def = ImmediateTensorDefinition(dataType: program.dataType,
                                                shape: param.shape,
                                                value: .int(i))

            case let .constant(.float(f)):
                def = ImmediateTensorDefinition(dataType: program.dataType,
                                                shape: param.shape,
                                                value: .float(f))
                
            case let .random(.int(i1), .int(i2)):
                def = RandomizingTensorDefinition(
                    dataType: program.dataType, shape: param.shape,
                    lowerBound: .int(i1), upperBound: .int(i2)
                )

            case let .random(.float(f1), .float(f2)):
                def = RandomizingTensorDefinition(
                    dataType: program.dataType, shape: param.shape,
                    lowerBound: .float(f1), upperBound: .float(f2)
                )

            default:
                preconditionFailure("This should not have passed Sema")
            }

            let variable = builder.declareTensor(def, named: param.name)
            environment[variable.name] = variable
        }

        /// Entry block
        let initBB = builder.makeBasicBlock(named: "entry")
        builder.module.entryBlock = initBB

        /// Generate hidden layers
        for layer in program.layers {
            let variable = build(layer.expression, named: layer.name)
            environment[layer.name] = variable
        }

        /// Generate output layer
        let outputVar = build(program.output.expression, named: program.output.name)
        environment[program.output.name] = outputVar
        
        builder.makeOutput(outputVar)

        /// Done!
        return builder.module
    }

    /// TODO: support recurrence
    @discardableResult
    func build(_ expression: Expression, named name: String? = nil) -> TensorVariable {
        switch expression {
        case let .call("sigmoid", args):
            let argOp = build(args[0])
            return builder.makeActivation(.sigmoid, argOp, name: name)
        case let .call("tanh", args):
            let argOp = build(args[0])
            return builder.makeActivation(.tanh, argOp, name: name)
        case let .call("relu", args):
            let argOp = build(args[0])
            return builder.makeActivation(.relu, argOp, name: name)
        case let .call("log", args):
            let argOp = build(args[0])
            return builder.makeTransformation(.log, argOp, name: name)
        case let .call("softmax", args):
            let argOp = build(args[0])
            return builder.makeTransformation(.softmax, argOp, name: name)
        case let .variable(variable):
            guard let op = environment[variable.name] as? TensorVariable else {
                preconditionFailure("Undeclared variable \(variable.name). This shouldn't have passed Sema.")
            }
            return op
        case let .infixOp(op, .constant(c), rhs):
            let lhsOp = c.operand(for: program.dataType)
            let rhsOp = build(rhs)
            return builder.makeBinaryOperation(op.instructionOperator,
                                               lhsOp, rhsOp, name: name)
        case let .infixOp(op, lhs, .constant(c)):
            let lhsOp = build(lhs)
            let rhsOp = c.operand(for: program.dataType)
            return builder.makeBinaryOperation(op.instructionOperator,
                                               lhsOp, rhsOp, name: name)
        case let .infixOp(op, lhs, rhs):
            let lhsOp = build(lhs)
            let rhsOp = build(rhs)
            return builder.makeBinaryOperation(op.instructionOperator,
                                               lhsOp, rhsOp, name: name)
        case let .negate(expr):
            let exprOp = build(expr)
            return builder.makeBinaryOperation(.sub, ImmediateOperand.int(0),
                                               exprOp, name: name)
        case let .product(lhs, rhs):
            let lhsOp = build(lhs)
            let rhsOp = build(rhs)
            return builder.makeProduct(lhsOp, rhsOp, name: name)
        case let .concat(exprs, dimension: dim):
            let exprOps = exprs.map { self.build($0) }
            return builder.makeConcatenation(exprOps, dimension: dim, name: name)
        case let .reshape(expr, shape: dims):
            let exprOp = build(expr)
            let targetShape = TensorShape(dims)
            precondition(exprOp.shape.contiguousSize == targetShape.contiguousSize,
                         "Tensor shape cast mismatch. This shouldn't have passed Sema.")
            return builder.makeShapeCast(exprOp, shape: targetShape, name: name)
        default:
            preconditionFailure("Unsupported expression \(expression). This shouldn't have passed Sema.")
        }
    }

}

fileprivate extension Constant {
    func operand(for dataType: DataType) -> ImmediateOperand {
        switch self {
        case let .int(i) where dataType.isFloat: return .float(Double(i))
        case let .int(i): return .int(i)
        case let .float(f): return .float(f)
        }
    }
}

fileprivate extension Expression.InfixOperator {
    var instructionOperator: Instruction.ArithmeticOperator {
        switch self {
        case .add: return .add
        case .sub: return .sub
        case .mul: return .mul
        case .div: return .div
        }
    }
}
