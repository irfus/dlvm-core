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
    var variables: [String : DLVM.ValueRepresentation] = [:]
    
    subscript(key: String) -> DLVM.ValueRepresentation? {
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
            let value = builder.declareInput(name: input.name, type: program.dataType, shape: input.shape)
            environment[value.name] = value
        }
        /// Define globals
        for param in program.parameters {
            let initializer: TensorInitializer
            switch param.initializer {
            case let .constant(.int(i)):
                initializer = .repeating(ImmediateValue(type: program.dataType, immediate: .int(i)))
            case let .constant(.float(f)):
                initializer = .repeating(ImmediateValue(type: program.dataType, immediate: .float(f)))
            case let .random(.int(i1), .int(i2)):
                initializer = .random(from: ImmediateValue(type: program.dataType, immediate: .int(i1)),
                                      to: ImmediateValue(type: program.dataType, immediate: .int(i2)))
            case let .random(.float(f1), .float(f2)):
                initializer = .random(from: ImmediateValue(type: program.dataType, immediate: .float(f1)),
                                      to: ImmediateValue(type: program.dataType, immediate: .float(f2)))
            default:
                preconditionFailure("This should not have passed Sema")
            }
            let value = builder.declareParameter(name: param.name, type: program.dataType,
                                                 shape: param.shape, initializer: initializer)
            environment[value.name] = value
        }
        
        /// Entry block
        builder.makeGlobalBasicBlock(named: "entry")

        /// Generate hidden layers
        for layer in program.layers {
            let value = build(layer.expression, named: layer.name)
            if layer.isOutput {
                let output = builder.declareOutput(name: layer.name, type: program.dataType,
                                                   shape: layer.shape)
                environment[output.name] = output
                builder.makeExport(value, to: output)
            }
            environment[layer.name] = value
        }

        /// Done!
        return builder.module
    }
    
    /// TODO: support recurrence
    @discardableResult
    func build(_ expression: Expression, named name: String? = nil) -> DLVM.Value {
        switch expression {
        case let .constant(c, _):
            return c.operand(for: program.dataType)
        case let .call(funcName, args, _):
            guard let function = builtinFunctionTable[funcName] else {
                preconditionFailure("Unknown function name. This shouldn't have passed Sema.")
            }
            let argOps = args.map { [unowned self] in self.build($0) }
            return function.makeInstruction(withArguments: argOps, using: builder, name: name)
        case let .variable(variable, _):
            guard let op = environment[variable.name] else {
                preconditionFailure("Undeclared variable \(variable.name). This shouldn't have passed Sema.")
            }
            if let input = op as? DLVM.Input {
                return builder.makeLoad(input)
            }
            if let val = op as? DLVM.Value {
                return val
            }
            /// Can't be an output. This won't be reached
            preconditionFailure("Unsupported expression \(expression)")
        case let .infixOp(op, lhs, rhs, _):
            let lhsOp = build(lhs), rhsOp = build(rhs)
            return builder.makeArithmeticOperation(op.instructionOperator,
                                                   lhsOp, rhsOp, name: name)
        case let .negate(expr, _):
            let exprOp = build(expr)
            return builder.makeElementwiseTransformation(.neg, exprOp)
        case let .product(lhs, rhs, _):
            let lhsOp = build(lhs), rhsOp = build(rhs)
            return builder.makeMatrixMultiplication(lhsOp, rhsOp, name: name)
        case let .concat(exprs, dimension: dim, _):
            let exprOps = exprs.map { [unowned self] in self.build($0) }
            return builder.makeConcatenation(exprOps, axis: dim, name: name)
        case let .reshape(expr, shape: dims, _):
            let exprOp = build(expr)
            let targetShape = TensorShape(dims)
            return builder.makeShapeCast(exprOp, targetShape: targetShape, name: name)
        case let .transpose(expr, _):
            let exprOp = build(expr)
            guard let targetShape = exprOp.shape.transpose else { fallthrough }
            return builder.makeShapeCast(exprOp, targetShape: targetShape, name: name)
        default:
            preconditionFailure("Unsupported expression \(expression). This shouldn't have passed Sema.")
        }
    }
    
}

fileprivate extension Constant {
    func operand(for dataType: DataType) -> ImmediateValue {
        switch self {
        case let .int(i, _) where dataType.base == .float:
            return ImmediateValue(type: dataType, immediate: .float(Double(i)))
        case let .int(i, _): return ImmediateValue(type: dataType, immediate: .int(i))
        case let .float(f, _): return ImmediateValue(type: dataType, immediate: .float(f))
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
