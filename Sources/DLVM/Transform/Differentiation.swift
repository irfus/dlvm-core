//
//  Differentiation.swift
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

/// WIP: Richard
/// - TODO: Refactor bigly

/// Replace every `gradient` instruction to a `call` to a function that
/// produces the gradient
open class Differentiation: TransformPass {
    public typealias Body = Module
    
    open class func run(on module: Module) throws -> Bool {
        var changed = false
        /// Run analysis before the transformation
        /// Only to check if this pass has been previously run
        let globalGradInfo = try module.analysis(from: GradientRelationAnalysis.self)

        var expanded: [Function : Function] = [:]

        for function in module {
            for instruction in function.instructions {
                if case let .gradient(.function(_, funcToDiff),
                                      from: diffIndex,
                                      wrt: varIndices,
                                      keeping: outputIndices) = instruction.kind,
                   funcToDiff.isDifferentiable {
                    if let _ = globalGradInfo.gradient(of: function,
                                                       from: diffIndex,
                                                       wrt: varIndices,
                                                       keepingOutputs: outputIndices) {
                        continue
                    }
                }
            }
        }

            /// If function is not differentiable, do nothing
//            guard function.isDifferentiable else { continue }
//            /// If gradient function exists, do nothing
//            if let _ = globalGradInfo.gradient(of: function) { continue }
//            /// Clone function
////            let grad = function.makeClone(named: func)
//            /// Expand this function
//            try expand(function)
//            changed = true
        return changed
    }
}

/// - Note: A new approach of adjoint code generation will be used.
/// Function will first be cloned and then adjoint code gets generated in
/// a real adjoint fasion

fileprivate extension Type {
    func makeLiteral(_ number: IntegerLiteralType) -> Use {
        switch canonical {
        case let .tensor(shape, type):
            return .tensor(shape, type, repeating: number)
        case let .tuple(subtypes):
            let sublits = subtypes.map{$0.makeLiteral(number)}
            return .literal(self, .tuple(sublits))
        case let .array(i, subtype):
            let sublit = subtype.makeLiteral(number)
            let sublits = Array(repeating: sublit, count: i)
            return .literal(self, .array(sublits))
        case _:
            fatalError()
        }
    }
}

/// Integer literal builder based on the unsafe (but convenient)
/// assumption that the value is a tensor
fileprivate extension Value {
    func makeTensor(repeating repeatedValue: IntegerLiteralType) -> Use {
        let canType = type.canonical
        guard case let .tensor(shape, dtype) = canType else {
            preconditionFailure("\(self), a.k.a. \(canType), is not tensor")
        }
        return .tensor(shape, dtype, repeating: repeatedValue)
    }

    func makeScalar(_ value: IntegerLiteralType) -> Use {
        let canType = type.canonical
        guard case let .tensor(_, dtype) = canType else {
            preconditionFailure("\(self), a.k.a. \(canType), is not tensor")
        }
        return .tensor(.scalar, dtype, repeating: value)
    }
}

/// Integer literal builder based on the unsafe (but convenient)
/// assumption that the value is a tensor
fileprivate extension Use {
    func makeTensor(repeating repeatedValue: IntegerLiteralType) -> Use {
        return value.makeTensor(repeating: repeatedValue)
    }

    func makeScalar(_ value: IntegerLiteralType) -> Use {
        return self.value.makeScalar(value)
    }
}

fileprivate class ADContext {
    var blocks: [BasicBlock : BasicBlock] = [:]
    var adjoints: [AnyHashable : Use] = [:]
    var clones: [AnyHashable : Use] = [:]

    unowned let forward: Function
    unowned let gradient: Function
    lazy var builder: IRBuilder = IRBuilder(function: self.gradient)

    init(forward: Function, gradient: Function) {
        self.forward = forward
        self.gradient = gradient
    }
}

fileprivate extension Differentiation {

    static func expand(_ function: Function) throws {
        let builder = IRBuilder(module: function.parent)
        /// Seed on return instructions
        let exits = try function.premise().exits
        for (block, returnInst) in exits {
            let retVal = returnInst.operands[0]
            let seed = retVal.type.makeLiteral(1)
//            context.insertAdjoint(seed, for: retVal)

            /// Differentiate
            for inst in block.reversed().dropFirst() {
//                differentiate(inst, using: builder, in: context)
            }

//            let tupleLit = LiteralValue(type: .tuple(function.arguments.map{$0.type}),
//                                        literal: .tuple(gradients))
//            builder.return(tupleLit.makeUse())
        }

    }

    static func differentiate(_ instruction: Instruction,
                              using bd: IRBuilder,
                              in context: ADContext) {
        guard let adjoint = context.adjoints[instruction] else {
            fatalError("Adjoint seed not found in AD")
        }
        var grad: [(operand: Use, derivative: Use)]
        switch instruction.kind {
        /* Basic arithmetic */
        case let .binary(.associative(.arithmetic(.add)), lhs, rhs, _):
            grad = [
                (lhs, adjoint), /// ∂f/∂x = D
                (rhs, adjoint), /// ∂f/∂y = D
            ]
        case let .binary(.associative(.arithmetic(.subtract)), lhs, rhs, bc):
            grad = [
                (lhs, adjoint),                                     /// ∂f/∂x = D
                (rhs, %bd.subtract(adjoint.makeScalar(0), adjoint, broadcasting: bc)), /// ∂f/∂y = -D
            ]
        case let .binary(.associative(.arithmetic(.multiply)), lhs, rhs, _):
            grad = [
                (lhs, rhs), /// ∂f/∂x = y
                (rhs, lhs), /// ∂f/∂y = x
            ]
        case let .binary(.associative(.arithmetic(.divide)), lhs, rhs, _):
            let lhsClone = lhs
            let rhsClone = rhs
            grad = [
                (lhs, %bd.divide(adjoint, rhsClone)),  /// ∂f/∂x = D/y
                (rhs, %bd.subtract(lhsClone.makeScalar(0), /// ∂f/∂y = -x/y^2
                                   %bd.divide(lhsClone, %bd.multiply(rhsClone, rhsClone))))
            ]

        /* Matrix multiplication */
        case let .matrixMultiply(lhs, rhs):
            let lhsClone = lhs
            let rhsClone = rhs
            grad = [
                /// ∂f/∂x = D • y^T
                (lhs, %bd.matrixMultiply(%bd.transpose(lhsClone), adjoint)),
                /// ∂f/∂y = x^T • D
                (rhs, %bd.matrixMultiply(adjoint, %bd.transpose(rhsClone))),
            ]


        /* Unary elementwise transformations */
        case let .unary(.exp, x):
            let cloned = instruction.makeUse()
            grad = [
                (x, cloned)
            ]

        case let .unary(.tanh, x):
            let cloned = instruction.makeUse()
            grad = [
                (x, %bd.subtract(cloned, %bd.subtract(x.makeScalar(1),
                                                      %bd.multiply(cloned, cloned)), broadcasting: [0]))
            ]

        case let .extract(from: x, at: _):
            grad = [
                (x, x.type.makeLiteral(1))
            ]

        default:
            /// - TODO: Implement all cases!
            fatalError("Unimplemented \(instruction)")
        }
    }

}
