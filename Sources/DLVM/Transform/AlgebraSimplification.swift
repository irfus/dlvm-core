//
//  AlgebraSimplification.swift
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

/// Algebra Simplification simplifies the following expressions
/// 1. Arithmetics
///    - x + 0 | 0 + x | x - 0 | x * 1 | 1 * x | x / 1 => x
///    - x * 0 | 0 * x => 0
///    - x^(-1) => 1 / x
///    - x^0 => 1
///    - x^1 => x
///    - x^2 => x * x
/// 2. Trignometry
///    - (e^x - e^(-x)) / 2 => sinh(x)
///    - (e^x + e^(-x)) / 2 => cosh(x)
///    - (e^x - e^(-x)) / (e^x + e^(-x)) => tanh(x)
///    - sin(0) | sinh(0) => 0
///    - cos(0) | cosh(0) => 1
///    - tan(0) | tanh(0) => 0
/// 3. Reassociation
///    - (e^x)^y => e^(x*y)
///    - e^x / e^y => e^(x-y)
///    - (x / y) / (z / a) => (x * a) / (y * z)
///    - (x / y) / z => x / (y * z)
///    - x / (y / z) => (x * z) / y
/// 4. Linear algebra
///    - (A^T)^T => A
open class AlgebraSimplification : TransformPass {
    public typealias Body = Function

    @discardableResult
    private static func performSimplification(on expr: AlgebraicExpression,
                                              in function: Function,
                                              using builder: IRBuilder,
                                              workList: inout [AlgebraicExpression]) -> Bool {
        /// Pattern-match expressions
        switch expr {
        // MARK: - 1. Arithmetics
        /// - x + 0 | 0 + x | x - 0 | x * 1 | 1 * x => x
        case let .zipWith(.associative(.add), x, 0, inst),
             let .zipWith(.associative(.add), 0, x, inst),
             let .zipWith(.associative(.subtract), x, 0, inst),
             let .zipWith(.associative(.multiply), x, 1, inst),
             let .zipWith(.associative(.multiply), 1, x, inst),
             let .zipWith(.associative(.divide), x, 1, inst):
            function.replaceAllUses(of: inst, with: %x)
            expr.removeIntermediates(upTo: x)
            workList.append(x)
        /// - x * 0 | 0 * x => 0
        case let .zipWith(.associative(.multiply), x, 0, inst),
             let .zipWith(.associative(.multiply), 0, x, inst):
            function.replaceAllUses(of: inst, with: %x.makeScalar(0))
            expr.removeIntermediates(upTo: x)
        /// - x^(-1) => 1 / x
        case let .zipWith(.associative(.power), x, -1, inst):
            builder.move(after: inst)
            let one = x.makeScalar(1)
            let div = builder.divide(%one, %x)
            function.replaceAllUses(of: inst, with: %div)
            expr.removeIntermediates(upTo: x)
            workList.append(.zipWith(.associative(.multiply), .atom(%one), x, div))
        /// - x^0 => 1
        case let .zipWith(.associative(.power), x, 0, inst):
            let newVal = expr.makeScalar(1)
            function.replaceAllUses(of: inst, with: %newVal)
            expr.removeIntermediates(upTo: x)
        /// - x^1 => x
        case let .zipWith(.associative(.power), x, 1, inst):
            function.replaceAllUses(of: inst, with: %x)
            expr.removeIntermediates(upTo: x)
            workList.append(x)
        /// - x^2 => x * x
        case let .zipWith(.associative(.power), x, 2, inst):
            builder.move(after: inst)
            let product = builder.multiply(%x, %x)
            function.replaceAllUses(of: inst, with: %product)
            expr.removeIntermediates(upTo: x)
            workList.append(.zipWith(.associative(.multiply), x, x, product))

        // MARK: - 2. Trignometry
        /// - (e^x - e^(-x)) / 2 => sinh(x)
        case let .zipWith(.associative(.divide),
                          .zipWith(.associative(.subtract),
                                   .map(.exp, x1, _),
                                   .map(.exp, .map(.negate, x2, _), _),
                                   _), 2, inst) where x1 == x2:
            builder.move(after: inst)
            let sinh = builder.buildInstruction(.map(.sinh, %x1))
            function.replaceAllUses(of: inst, with: %sinh)
            expr.removeIntermediates(upTo: x1)
            workList.append(.map(.sinh, x1, sinh))
        /// - (e^x - e^(-x)) / 2 => cosh(x)
        case let .zipWith(.associative(.divide),
                          .zipWith(.associative(.add),
                                   .map(.exp, x1, _),
                                   .map(.exp, .map(.negate, x2, _), _),
                                   _), 2, inst) where x1 == x2:
            builder.move(after: inst)
            let cosh = builder.buildInstruction(.map(.cosh, %x1))
            function.replaceAllUses(of: inst, with: %cosh)
            expr.removeIntermediates(upTo: x1)
            workList.append(.map(.cosh, x1, cosh))
        /// sin(0), sinh(0), tan(0), tanh(0)
        case let .map(.sin, 0, inst),
             let .map(.sinh, 0, inst),
             let .map(.tan, 0, inst),
             let .map(.tanh, 0, inst):
            function.replaceAllUses(of: inst, with: %expr.makeScalar(0))
            expr.removeIntermediates()
        /// cos(0), cosh(0)
        case let .map(.cos, 0, inst),
             let .map(.cosh, 0, inst):
            function.replaceAllUses(of: inst, with: %expr.makeScalar(1))
            expr.removeIntermediates()

        // MARK: - 3. Reassociation
        /// - (e^x)^y => e^(x*y)
        case let .zipWith(.associative(.power), .map(.exp, x, _), y, inst):
            builder.move(after: inst)
            let mul = builder.multiply(%x, %y)
            let exp = builder.exp(%mul)
            function.replaceAllUses(of: inst, with: %exp)
            expr.removeIntermediates(upTo: x, y)
            workList.append(.map(.exp, .zipWith(.associative(.multiply), x, y, mul), exp))
        /// - e^x / e^y => e^(x-y)
        case let .zipWith(.associative(.divide), .map(.exp, x, _), .map(.exp, y, _), inst):
            builder.move(after: inst)
            let sub = builder.subtract(%x, %y)
            let exp = builder.exp(%sub)
            function.replaceAllUses(of: inst, with: %exp)
            expr.removeIntermediates(upTo: x, y)
            workList.append(.map(.exp, .zipWith(.associative(.subtract), x, y, sub), exp))
        /// - x / e^y => x * e^(-y)
        case let .zipWith(.associative(.divide), x, .map(.exp, y, _), inst):
            builder.move(after: inst)
            let neg = builder.map(.negate, %y)
            let exp = builder.exp(%neg)
            let mul = builder.multiply(%x, %exp)
            function.replaceAllUses(of: inst, with: %mul)
            expr.removeIntermediates(upTo: x, y)
            workList.append(.zipWith(.associative(.multiply), x,
                                     .map(.exp, .map(.negate, y, neg), exp), mul))
        /// - x / y^z => x * y^(-z)
        case let .zipWith(.associative(.divide), x, .zipWith(.associative(.power), y, z, _), inst):
            builder.move(after: inst)
            let neg = builder.map(.negate, %z)
            let pwr = builder.power(%y, %neg)
            let mul = builder.multiply(%x, %pwr)
            function.replaceAllUses(of: inst, with: %mul)
            expr.removeIntermediates(upTo: x, y, z)
            workList.append(.zipWith(.associative(.multiply), x,
                                     .zipWith(.associative(.power), y,
                                              .map(.negate, z, neg), pwr), mul))
        /// - (x / y) / (z / a) => (x * a) / (y * z)
        case let .zipWith(.associative(.divide),
                          .zipWith(.associative(.divide), x, y, _),
                          .zipWith(.associative(.divide), z, a, _), inst):
            builder.move(after: inst)
            let mulL = builder.multiply(%x, %a)
            let mulR = builder.multiply(%y, %z)
            let div = builder.divide(%mulL, %mulR)
            function.replaceAllUses(of: inst, with: %div)
            expr.removeIntermediates(upTo: x, y, z, a)
            workList.append(.zipWith(.associative(.divide),
                                     .zipWith(.associative(.multiply), x, a, mulL),
                                     .zipWith(.associative(.multiply), y, z, mulR),
                                     div))
        /// - (x / y) / z => x / (y * z)
        case let .zipWith(.associative(.divide),
                          .zipWith(.associative(.divide), x, y, _),
                          z, inst):
            builder.move(after: inst)
            let mul = builder.multiply(%y, %z)
            let div = builder.divide(%x, %mul)
            function.replaceAllUses(of: inst, with: %div)
            expr.removeIntermediates(upTo: x, y, z)
            workList.append(.zipWith(.associative(.divide), x,
                                     .zipWith(.associative(.multiply), y, z, mul),
                                     div))
        /// - x / (y / z) => (x * z) / y
        case let .zipWith(.associative(.divide), x, .zipWith(.associative(.divide), y, z, _), inst):
            builder.move(after: inst)
            let mul = builder.multiply(%x, %z)
            let div = builder.divide(%mul, %y)
            function.replaceAllUses(of: inst, with: %div)
            expr.removeIntermediates(upTo: x, y, z)
            workList.append(.zipWith(.associative(.divide),
                                     .zipWith(.associative(.multiply), x, z, mul), y, div))

        // MARK: - 4. Linear Algebra
        /// - (A^T)^T => A
        case let .transpose(.transpose(A, _), inst):
            builder.move(after: inst)
            function.replaceAllUses(of: inst, with: %A)
            expr.removeIntermediates(upTo: A)
            workList.append(A)

        default:
            return false
        }
        return true
    }

    // MARK: - Pass entry

    open class func run(on body: Function) throws -> Bool {
        var changed = false
        var workList: [AlgebraicExpression] = []
        let builder = IRBuilder(function: body)
        /// First iteration
        for bb in body {
            let algExprs = try bb.analysis(from: AlgebraicExpressionAnalysis.self)
            for expr in algExprs {
                workList.append(expr)
            }
        }
        /// Iterate through the worklist and optimize them
        while let expr = workList.popLast() {
            for expr in expr.transposeTraversed(in: .breadthFirst) where !expr.isAtom {
                let newlyChanged = performSimplification(on: expr, in: body, using: builder, workList: &workList)
                changed = changed || newlyChanged
                if newlyChanged { break }
            }
        }
        return changed
    }
}
