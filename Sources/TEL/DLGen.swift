//
//  DLGen.swift
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
        /// Define globals
//        for param in program.parameters {
//            let initializer: TensorLiteral
//            switch param.initializer {
//            case let .constant(.int(i)):
//                initializer = .repeating(.int(i))
//            case let .constant(.float(f)):
//                initializer = .repeating(.float(f))
//            case let .random(.int(i1), .int(i2)):
//                initializer = .random(from: .int(i1), to: .int(i2))
//            case let .random(.float(f1), .float(f2)):
//                initializer = .random(from: .float(f1), to: .float(f2))
//            default:
//                preconditionFailure("This should not have passed Sema")
//            }
//            let variable = GlobalValue(name: param.name,
//                                       kind: .variable,
//                                       type: .tensor(param.shape, program.dataType),
//                                       initializer: .tensor(initializer))
//            let use = builder.buildGlobalValue(variable)
//            environment[param.name] = use
//        }

        /// Build pure inference function
        var args: [(String, Type)] = []
        for input in program.inputs {
            let arg: Type = .box(.tensor(input.shape, program.dataType), .compute)
            args.append((input.name, arg))
        }
        
        /// Generate a tuple for parameters
        let paramTupleType = builder.buildAlias(
            named: "\(program.moduleName)_params",
            for: .tuple(program.parameters.map {
                .box(.tensor($0.shape, self.program.dataType), .compute)
            })
        )
        args.append(("params", paramTupleType))

        let output = program.layers.first(where: {$0.isOutput})! // BAD!
        /// Compute function
        let function = builder.buildFunction(named: "inference", arguments: args,
                                             result: .box(.tensor(output.shape, program.dataType), .compute),
                                             attributes: [ .differentiable, .compute ])
        builder.move(to: function.entry)

        /// Add all args into env
        for argDef in function.arguments {
            let use = Use.argument(argDef.type, argDef)
            environment[argDef.name] = use
        }

        /// Emit tuple extraction
        let paramsVal = function.argumentValue(named: "params")!
        for (i, param) in program.parameters.enumerated() {
            let val = builder.buildInstruction(.extract(from: paramsVal, at: [.index(i)]))
            environment[param.name] = val
        }

        /// Generate hidden layers
        var result: Use?
        for layer in program.layers {
            let op = build(layer.expression, named: layer.name)
            result = op
            environment[layer.name] = op
        }
        builder.buildInstruction(.return(result))

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
            preconditionFailure("Unknown variable name \(variable.name). Something's wrong with DLGen")
        case let .call(funcName, args, _):
            guard let opKind = intrinsicTable[funcName] else {
                preconditionFailure("Unknown function name. This shouldn't have passed Sema.")
            }
            let argOps = args.map { [unowned self] in self.build($0) }
            let kind = try! opKind.makeInstruction(with: argOps)
            return builder.buildInstruction(kind, name: name)
        case let .infixOp(op, lhs, rhs, _):
            let lhsOp = build(lhs), rhsOp = build(rhs)
            let operation: InstructionKind = .zipWith(.associative(.arithmetic(op.instructionOperator)), lhsOp, rhsOp)
            return builder.buildInstruction(operation, name: name)
        case let .negate(expr, _):
            let exprOp = build(expr)
            let operation: InstructionKind = .map(.elementwise(.neg), exprOp)
            return builder.buildInstruction(operation, name: name)
        case let .product(lhs, rhs, _):
            let lhsOp = build(lhs), rhsOp = build(rhs)
            let operation: InstructionKind = .matrixMultiply(lhsOp, rhsOp)
            return builder.buildInstruction(operation, name: name)
        case let .concat(exprs, dimension: dim, _):
            let exprOps = exprs.map { [unowned self] in self.build($0) }
            let operation: InstructionKind = .concatenate(exprOps, axis: dim)
            return builder.buildInstruction(operation, name: name)
        case let .reshape(expr, shape: dims, _):
            let exprOp = build(expr)
            let targetShape = TensorShape(dims)
            let operation: InstructionKind = .shapeCast(exprOp, targetShape)
            return builder.buildInstruction(operation, name: name)
        case let .transpose(expr, _):
            let exprOp = build(expr)
            return builder.buildInstruction(.transpose(exprOp), name: name)
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
        let litVal = LiteralValue(type: .tensor(.scalar, dataType), literal: literal)
        return .literal(litVal)
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
