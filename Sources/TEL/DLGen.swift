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
    var variables: [String : DLVM.Use] = [:]

    var outputs: [String : DLVM.Def<Output>] = [:]
    var placeholders: [String : DLVM.Def<Placeholder>] = [:]
    weak var endBlock: BasicBlock?

    func containsValue(named name: String) -> Bool {
        return variables.keys.contains(name)
    }

    subscript(key: String) -> DLVM.Use {
        get {
            return variables[key]!
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
            let placeholder = Placeholder(shape: input.shape,
                                          type: program.dataType,
                                          isRecurrent: false)
            let ph = builder.declare(placeholder, name: input.name)
            environment.placeholders[input.name] = ph
        }
        /// Define globals
        for param in program.parameters {
            let initializer: TensorLiteral
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
            let variable = GlobalValue(kind: .variable,
                                       shape: param.shape, type: program.dataType,
                                       initializer: .tensor(initializer))
            let use = builder.declare(variable, name: param.name)
            environment[param.name] = use
        }

        /// Build pure inference function
        var args: [(String, DLVM.Argument)] = []
        for input in program.inputs {
            let arg = Argument(type: .tensor(input.shape, program.dataType))
            args.append((input.name, arg))
        }
        for param in program.parameters {
            let arg = Argument(type: .tensor(param.shape, program.dataType))
            args.append((param.name, arg))
        }
        let output = program.layers.first(where: {$0.isOutput})! // BAD!
        let function = builder.buildFunction(named: "inference", arguments: args,
                                             result: .tensor(output.shape, program.dataType),
                                             isDifferentiable: true)
        builder.move(to: function.entry)

        /// Add all args into env
        for argDef in function.arguments {
            let use = Use(kind: .argument(argDef))
            environment[argDef.name] = use
        }

        /// Generate hidden layers
        var result: Use?
        for layer in program.layers {
            let op = build(layer.expression, named: layer.name)
            result = op
            environment[layer.name] = op
        }
        builder.buildControl(.`return`(result))


        /// Main function
//        let function = builder.buildFunction(named: "main", arguments: [], result: nil, isDifferentiable: false)
//        /// Entry block
//        builder.move(to: function.entry)
//        /// Load all inputs in and call pure inference function
//        builder.buildFunction(named: "inference",
//                              arguments: <#T##[(String, Argument)]#>, result: <#T##Argument?#>, isDifferentiable: <#T##Bool#>)
        
//            if layer.isOutput {
//                let out = Output(shape: layer.shape, type: program.dataType, isRecurrent: false)
//                let def = builder.declare(out, name: layer.name)
//                builder.buildControl(.yield(op, to: def))
//            }
        
        return builder.module
    }
    
    /// TODO: support recurrence
    @discardableResult
    func build(_ expression: Expression, named name: String? = nil) -> DLVM.Use {
        switch expression {
        case let .constant(c, _):
            return c.operand(for: program.dataType)
        case let .variable(variable, _):
            if environment.containsValue(named: variable.name) {
                return environment[variable.name]
            }
            preconditionFailure("Unknown variable name. Something's wrong with DLGen")
        case let .call(funcName, args, _):
            guard let opKind = intrinsicTable[funcName] else {
                preconditionFailure("Unknown function name. This shouldn't have passed Sema.")
            }
            let argOps = args.map { [unowned self] in self.build($0) }
            let operation = try! opKind.makeOperation(with: argOps)
            return builder.buildOperation(operation, name: name)
        case let .infixOp(op, lhs, rhs, _):
            let lhsOp = build(lhs), rhsOp = build(rhs)
            let operation: Operation = .binary(.associative(.arithmetic(op.instructionOperator)), lhsOp, rhsOp)
            return builder.buildOperation(operation, name: name)
        case let .negate(expr, _):
            let exprOp = build(expr)
            let operation: Operation = .unary(.elementwise(.neg), exprOp)
            return builder.buildOperation(operation, name: name)
        case let .product(lhs, rhs, _):
            let lhsOp = build(lhs), rhsOp = build(rhs)
            let operation: Operation = .matrixMultiply(lhsOp, rhsOp)
            return builder.buildOperation(operation, name: name)
        case let .concat(exprs, dimension: dim, _):
            let exprOps = exprs.map { [unowned self] in self.build($0) }
            let operation: Operation = .concatenate(exprOps, axis: dim)
            return builder.buildOperation(operation, name: name)
        case let .reshape(expr, shape: dims, _):
            let exprOp = build(expr)
            let targetShape = TensorShape(dims)
            let operation: Operation = .shapeCast(exprOp, targetShape)
            return builder.buildOperation(operation, name: name)
        case let .transpose(expr, _):
            let exprOp = build(expr)
            return builder.buildOperation(.transpose(exprOp), name: name)
        default:
            preconditionFailure("Unsupported expression \(expression). This shouldn't have passed Sema.")
        }
    }
    
}

fileprivate extension Constant {
    func operand(for dataType: DataType) -> DLVM.Use {
        let literal: Literal
        switch self {
        case let .int(i, _) where dataType.base == .float:
            literal = .scalar(.float(Double(i)))
        case let .int(i, _):
            literal = .scalar(.int(i))
        case let .float(f, _):
            literal = .scalar(.float(f))
        }
        return Use(kind: .literal(LiteralValue(shape: .scalar, dataType: dataType, literal: literal)))
    }
}

fileprivate extension Expression.InfixOperator {
    var instructionOperator: ArithmeticOp {
        switch self {
        case .add: return .add
        case .sub: return .subtract
        case .mul: return .multiply
        case .div: return .divide
        }
    }
}
