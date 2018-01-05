//
//  Passes.swift
//  DLCommandLineTools
//
//  Copyright 2016-2017 The DLVM Team.
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
import protocol Utility.StringEnumArgument
import enum Utility.ShellCompletion

public enum TransformPass: String {
    case Differentiation
    case DeadCodeElimination
    case CommonSubexpressionElimination
    case AlgebraSimplification
    case LinearAlgebraFusion
    case StackPromotion
    case ValuePromotion
    case MatrixMultiplicationReordering
    case LiteralBroadcastingPromotion
}

public extension TransformPass {
    var abbrevation: String {
        switch self {
        case .Differentiation: return "AD"
        case .DeadCodeElimination: return "DCE"
        case .CommonSubexpressionElimination: return "CSE"
        case .AlgebraSimplification: return "AS"
        case .LinearAlgebraFusion: return "LAF"
        case .StackPromotion: return "SP"
        case .ValuePromotion: return "VP"
        case .MatrixMultiplicationReordering: return "MMR"
        case .LiteralBroadcastingPromotion: return "LBP"
        }
    }

    var description: String {
        switch self {
        case .Differentiation: return "differentiation"
        case .DeadCodeElimination: return "dead code elimination"
        case .CommonSubexpressionElimination: return "common subexpression elimination"
        case .AlgebraSimplification: return "algebra simplification"
        case .LinearAlgebraFusion: return "linear algebra fusion"
        case .StackPromotion: return "stack promotion"
        case .ValuePromotion: return "value promotion"
        case .MatrixMultiplicationReordering: return "matrix multiplication reordering"
        case .LiteralBroadcastingPromotion: return "literal broadcasting promotion"
        }
    }

    // NOTE: Can be shortened with enum iteration
    init?(rawValue: String) {
        typealias T = TransformPass
        switch rawValue {
        case T.Differentiation.abbrevation, T.Differentiation.rawValue:
            self = .Differentiation
        case T.DeadCodeElimination.abbrevation, T.DeadCodeElimination.rawValue:
            self = .DeadCodeElimination
        case T.CommonSubexpressionElimination.abbrevation, T.CommonSubexpressionElimination.rawValue:
            self = .CommonSubexpressionElimination
        case T.AlgebraSimplification.abbrevation, T.AlgebraSimplification.rawValue:
            self = .AlgebraSimplification
        case T.LinearAlgebraFusion.abbrevation, T.LinearAlgebraFusion.rawValue:
            self = .LinearAlgebraFusion
        case T.StackPromotion.abbrevation, T.StackPromotion.rawValue:
            self = .StackPromotion
        case T.ValuePromotion.abbrevation, T.ValuePromotion.rawValue:
            self = .ValuePromotion
        case T.MatrixMultiplicationReordering.abbrevation, T.MatrixMultiplicationReordering.rawValue:
            self = .MatrixMultiplicationReordering
        case T.LiteralBroadcastingPromotion.abbrevation, T.LiteralBroadcastingPromotion.rawValue:
            self = .LiteralBroadcastingPromotion
        default:
            return nil
        }
    }
}

extension TransformPass : StringEnumArgument {
    public static var completion: ShellCompletion {
        // NOTE: Can be shortened with enum iteration
        return .values([
            (Differentiation.abbrevation, Differentiation.description),
            (DeadCodeElimination.abbrevation, DeadCodeElimination.description),
            (CommonSubexpressionElimination.abbrevation, CommonSubexpressionElimination.description),
            (AlgebraSimplification.abbrevation, AlgebraSimplification.description),
            (LinearAlgebraFusion.abbrevation, LinearAlgebraFusion.description),
            (StackPromotion.abbrevation, StackPromotion.description),
            (ValuePromotion.abbrevation, ValuePromotion.description),
            (MatrixMultiplicationReordering.abbrevation, MatrixMultiplicationReordering.description),
            (LiteralBroadcastingPromotion.abbrevation, LiteralBroadcastingPromotion.description)
        ])
    }
}

public func runPass(_ pass: TransformPass, on module: Module,
                bypassingVerification noVerify: Bool = false) throws {
    var changed: Bool
    switch pass {
    case .Differentiation:
        changed = module.applyTransform(Differentiation.self, bypassingVerification: noVerify)
    case .DeadCodeElimination:
        changed = module.mapTransform(DeadCodeElimination.self, bypassingVerification: noVerify)
    case .CommonSubexpressionElimination:
        changed = module.mapTransform(CommonSubexpressionElimination.self, bypassingVerification: noVerify)
    case .AlgebraSimplification:
        changed = module.mapTransform(AlgebraSimplification.self, bypassingVerification: noVerify)
    case .LinearAlgebraFusion:
        changed = module.mapTransform(LinearAlgebraFusion.self, bypassingVerification: noVerify)
    case .StackPromotion:
        changed = module.mapTransform(StackPromotion.self, bypassingVerification: noVerify)
    case .ValuePromotion:
        changed = module.mapTransform(ValuePromotion.self, bypassingVerification: noVerify)
    case .MatrixMultiplicationReordering:
        changed = module.mapTransform(MatrixMultiplicationReordering.self, bypassingVerification: noVerify)
    case .LiteralBroadcastingPromotion:
        changed = module.mapTransform(LiteralBroadcastingPromotion.self, bypassingVerification: noVerify)
    }
    print("\(pass.abbrevation):", changed ? "changed" : "unchanged")
}
