//
//  Passes.swift
//  DLCommandLineTools
//
//  Copyright 2016-2018 The DLVM Team.
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
    case algebraSimplification = "AlgebraSimplification"
    case cfgCanonicalization = "CFGCanonicalization"
    case cfgSimplification = "CFGSimpliciation"
    case commonSubexpressionElimination = "CommonSubexpressionElimination"
    case deadCodeElimination = "DeadCodeElimination"
    case differentiation = "Differentiation"
    case linearAlgebraFusion = "LinearAlgebraFusion"
    case literalBroadcastingPromotion = "LiteralBroadcastingPromotion"
    case matrixMultiplicationReordering = "MatrixMultiplicationReordering"
    case stackPromotion = "StackPromotion"
    case valuePromotion = "ValuePromotion"
}

public extension TransformPass {
    var abbrevation: String {
        switch self {
        case .algebraSimplification: return "AS"
        case .cfgCanonicalization: return "CFGCan"
        case .cfgSimplification: return "CFGSimp"
        case .commonSubexpressionElimination: return "CSE"
        case .deadCodeElimination: return "DCE"
        case .differentiation: return "AD"
        case .linearAlgebraFusion: return "LAF"
        case .literalBroadcastingPromotion: return "LBP"
        case .matrixMultiplicationReordering: return "MMR"
        case .stackPromotion: return "SP"
        case .valuePromotion: return "VP"
        }
    }

    var description: String {
        switch self {
        case .algebraSimplification: return "algebra simplification"
        case .cfgCanonicalization: return "CFG canonicalization"
        case .cfgSimplification: return "CFG simplification"
        case .commonSubexpressionElimination:
            return "common subexpression elimination"
        case .deadCodeElimination: return "dead code elimination"
        case .differentiation: return "differentiation"
        case .linearAlgebraFusion: return "linear algebra fusion"
        case .literalBroadcastingPromotion:
            return "literal broadcasting promotion"
        case .matrixMultiplicationReordering:
            return "matrix multiplication reordering"
        case .stackPromotion: return "stack promotion"
        case .valuePromotion: return "value promotion"
        }
    }

    // NOTE: Can be shortened with enum iteration
    init?(rawValue: String) {
        typealias T = TransformPass
        switch rawValue {
        case T.algebraSimplification.abbrevation,
             T.algebraSimplification.rawValue:
            self = .algebraSimplification
        case T.cfgCanonicalization.abbrevation,
             T.cfgCanonicalization.rawValue:
            self = .cfgCanonicalization
        case T.cfgSimplification.abbrevation,
             T.cfgSimplification.rawValue:
            self = .cfgSimplification
        case T.commonSubexpressionElimination.abbrevation,
             T.commonSubexpressionElimination.rawValue:
            self = .commonSubexpressionElimination
        case T.deadCodeElimination.abbrevation,
             T.deadCodeElimination.rawValue:
            self = .deadCodeElimination
        case T.differentiation.abbrevation,
             T.differentiation.rawValue:
            self = .differentiation
        case T.linearAlgebraFusion.abbrevation,
             T.linearAlgebraFusion.rawValue:
            self = .linearAlgebraFusion
        case T.literalBroadcastingPromotion.abbrevation,
             T.literalBroadcastingPromotion.rawValue:
            self = .literalBroadcastingPromotion
        case T.matrixMultiplicationReordering.abbrevation,
             T.matrixMultiplicationReordering.rawValue:
            self = .matrixMultiplicationReordering
        case T.stackPromotion.abbrevation,
             T.stackPromotion.rawValue:
            self = .stackPromotion
        case T.valuePromotion.abbrevation,
             T.valuePromotion.rawValue:
            self = .valuePromotion
        default:
            return nil
        }
    }
}

extension TransformPass : StringEnumArgument {
    public static var completion: ShellCompletion {
        // NOTE: Can be shortened with enum iteration
        return .values([
            (algebraSimplification.abbrevation,
             algebraSimplification.description),
            (cfgCanonicalization.abbrevation,
             cfgCanonicalization.description),
            (CFGSimplification.abbrevation,
             CFGSimplification.description),
            (commonSubexpressionElimination.abbrevation,
             commonSubexpressionElimination.description),
            (deadCodeElimination.abbrevation,
             deadCodeElimination.description),
            (differentiation.abbrevation,
             differentiation.description),
            (linearAlgebraFusion.abbrevation,
             linearAlgebraFusion.description),
            (literalBroadcastingPromotion.abbrevation,
             literalBroadcastingPromotion.description),
            (matrixMultiplicationReordering.abbrevation,
             matrixMultiplicationReordering.description),
            (stackPromotion.abbrevation,
             stackPromotion.description),
            (valuePromotion.abbrevation,
             valuePromotion.description)
        ])
    }
}

public func runPass(_ pass: TransformPass, on module: Module,
                    bypassingVerification noVerify: Bool = false) throws {
    var changed: Bool
    switch pass {
    case .algebraSimplification:
        changed = module.mapTransform(AlgebraSimplification.self,
                                      bypassingVerification: noVerify)
    case .cfgCanonicalization:
        changed = module.mapTransform(CFGCanonicalization.self,
                                      bypassingVerification: noVerify)
    case .cfgSimplification:
        changed = module.mapTransform(CFGSimplification.self,
                                      bypassingVerification: noVerify)
    case .commonSubexpressionElimination:
        changed = module.mapTransform(CommonSubexpressionElimination.self,
                                      bypassingVerification: noVerify)
    case .deadCodeElimination:
        changed = module.mapTransform(DeadCodeElimination.self,
                                      bypassingVerification: noVerify)
    case .differentiation:
        changed = module.applyTransform(Differentiation.self,
                                        bypassingVerification: noVerify)
    case .linearAlgebraFusion:
        changed = module.mapTransform(LinearAlgebraFusion.self,
                                      bypassingVerification: noVerify)
    case .literalBroadcastingPromotion:
        changed = module.mapTransform(LiteralBroadcastingPromotion.self,
                                      bypassingVerification: noVerify)
    case .matrixMultiplicationReordering:
        changed = module.mapTransform(MatrixMultiplicationReordering.self,
                                      bypassingVerification: noVerify)
    case .stackPromotion:
        changed = module.mapTransform(StackPromotion.self,
                                      bypassingVerification: noVerify)
    case .valuePromotion:
        changed = module.mapTransform(ValuePromotion.self,
                                      bypassingVerification: noVerify)
    }
    print("\(pass.abbrevation):", changed ? "changed" : "unchanged")
}
