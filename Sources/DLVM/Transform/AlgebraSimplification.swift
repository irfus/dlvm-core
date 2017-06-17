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

import Foundation

/// Algebra Simplification simplifies the following expressions
/// - x^0 => 1
/// - x^1 => x
/// - x^2 => x * x
/// - (e^x - e^(-x)) / 2 => sinh(x)
/// - (e^x + e^(-x)) / 2 => cosh(x)
/// - (e^x - e^(-x)) / (e^x + e^(-x)) => tanh(x)
/// - sin(0) | sinh(0) => 0
/// - cos(0) | cosh(0) => 1
/// - tan(0) | tanh(0) => 0
open class AlgebraSimplification : TransformPass {
    public typealias Body = Function

    @discardableResult
    private static func performSimplification(on expr: AlgebraicExpression,
                                              in function: Function,
                                              using builder: IRBuilder,
                                              workList: inout [AlgebraicExpression]) -> Bool {
        /// Pattern-match expressions
        switch expr {
        /// - x^0 => 1
        case let .zipWith(.associative(.power), x, 0, inst):
            let newVal = expr.makeLiteral(1)
            function.replaceAllUses(of: inst, with: %newVal)
            expr.removeIntermediates(upTo: x)
            workList.append(x)
        /// - x^1 => x
        case let .zipWith(.associative(.power), x, 1, inst):
            function.replaceAllUses(of: inst, with: x.value.makeUse())
            expr.removeIntermediates(upTo: x)
            workList.append(x)
        /// - x^2 => x * x
        case let .zipWith(.associative(.power), x, 2, inst):
            builder.move(after: inst)
            let product = builder.multiply(%x, %x)
            function.replaceAllUses(of: inst, with: %product)
            expr.removeIntermediates(upTo: x)
            workList.append(.zipWith(.associative(.multiply), x, x, product))
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
            builder.move(after: inst)
            expr.removeIntermediates()
            function.replaceAllUses(of: inst, with: %expr.makeLiteral(0))
        /// cos(0), cosh(0)
        case let .map(.cos, 0, inst),
             let .map(.cosh, 0, inst):
            builder.move(after: inst)
            expr.removeIntermediates()
            function.replaceAllUses(of: inst, with: %expr.makeLiteral(1))
        default:
            return false
        }
        return true
    }

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
