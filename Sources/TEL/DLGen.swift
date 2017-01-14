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
    var variables: [String : Value] = [:]
    
    subscript(key: String) -> Value? {
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
            let type = program.dataType.makeTensorType(with: input.shape)
            let value = builder.declareInput(name: input.name, type: type)
            environment[value.name] = value
        }
        /// Define globals
        for param in program.parameters {
            let initializer: TensorInitializer
            switch param.initializer {
            case let .constant(.int(i)):
                initializer = .repeating(.int(i))
            case let .constant(.float(f)):
                initializer = .repeating(.float(f))
            case let .random(.int(i1), .int(i2)):
                initializer = .random(from: .int(i1), to: .int(i2))
            case let .random(.float(f1), .float(f2)):
                initializer = .random(from: .float(f1), to: .float(f2))
            default:
                preconditionFailure("This should not have passed Sema")
            }
            let type = program.dataType.makeTensorType(with: param.shape)
            let value = builder.declareParameter(name: param.name,
                                                 type: type, initializer: initializer)
            environment[value.name] = value
        }
        
        /// Entry block
        let initBB = builder.makeBasicBlock(named: "entry")
        builder.module.entryBlock = initBB
        
        /// Generate hidden layers
        for layer in program.layers {
            let variable = build(layer.expression, named: layer.name)
            environment[layer.name] = variable
        }

        /// Done!
        return builder.module
    }
    
    /// TODO: support recurrence
    @discardableResult
    func build(_ expression: Expression, named name: String? = nil) -> Value {
        switch expression {
        case let .call("sigmoid", args):
            let argOp = build(args[0])
            return builder.makeElementwiseCall(.sigmoid, argOp)
        case let .call("tanh", args):
            let argOp = build(args[0])
            return builder.makeElementwiseCall(.sigmoid, argOp)
        case let .call("relu", args):
            let argOp = build(args[0])
            return builder.makeElementwiseCall(.sigmoid, argOp)
        case let .call("log", args):
            let argOp = build(args[0])
            return builder.makeElementwiseCall(.sigmoid, argOp)
        case let .call("softmax", args):
            let argOp = build(args[0])
            return builder.makeAggregateCall(.softmax, argOp)
        case let .variable(variable):
            guard let op = environment[variable.name] else {
                preconditionFailure("Undeclared variable \(variable.name). This shouldn't have passed Sema.")
            }
            return op
        case let .infixOp(op, .constant(c), rhs):
            let lhsOp = c.operand(for: program.dataType), rhsOp = build(rhs)
            return builder.makeArithmeticOperation(op.instructionOperator,
                                                   lhsOp, rhsOp, name: name)
        case let .infixOp(op, lhs, .constant(c)):
            let lhsOp = build(lhs), rhsOp = c.operand(for: program.dataType)
            return builder.makeArithmeticOperation(op.instructionOperator,
                                                   lhsOp, rhsOp, name: name)
        case let .infixOp(op, lhs, rhs):
            let lhsOp = build(lhs), rhsOp = build(rhs)
            return builder.makeArithmeticOperation(op.instructionOperator,
                                                   lhsOp, rhsOp, name: name)
        case let .negate(expr):
            let exprOp = build(expr)
            return builder.makeNegation(exprOp, name: name)
            
        case let .product(lhs, rhs):
            let lhsOp = build(lhs), rhsOp = build(rhs)
            return builder.makeTensorProduct(lhsOp, rhsOp, name: name)
        case let .concat(exprs, dimension: dim):
            let exprOps = exprs.map { self.build($0) }
            return builder.makeConcatenation(exprOps, axis: dim, name: name)
        case let .reshape(expr, shape: dims):
            let exprOp = build(expr)
            let targetShape = TensorShape(dims)
            return builder.makeShapeCast(exprOp, targetShape: targetShape, name: name)
        default:
            preconditionFailure("Unsupported expression \(expression). This shouldn't have passed Sema.")
        }
    }
    
}

fileprivate extension Constant {
    func operand(for dataType: DataType) -> ImmediateValue {
        switch self {
        case let .int(i) where dataType.base == .float:
            return ImmediateValue(type: dataType, immediate: .float(Double(i)))
        case let .int(i): return ImmediateValue(type: dataType, immediate: .int(i))
        case let .float(f): return ImmediateValue(type: dataType, immediate: .float(f))
        }
    }
}

fileprivate extension Expression.InfixOperator {
    var instructionOperator: ArithmeticOperator {
        switch self {
        case .add: return .add
        case .sub: return .subtract
        case .mul: return .multiply
        case .div: return .divide
        }
    }
}
