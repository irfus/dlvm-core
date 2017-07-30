//
//  Passes.swift
//  DLCommandLineTools
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

public func runPass(named name: String, on module: Module) throws {
    switch name {
    case "AD", "Differentiation":
        module.applyTransform(Differentiation.self)
    case "Can", "Canonicalization":
        module.applyTransform(Canonicalization.self)
    case "CP", "Checkpointing":
        module.mapTransform(Checkpointing.self)
    case "DCE", "DeadCodeElimination":
        module.mapTransform(DeadCodeElimination.self)
    case "CSE", "CommonSubexpressionElimination":
        module.mapTransform(CommonSubexpressionElimination.self)
    case "AS", "AlgebraSimplification":
        module.forEach { fn in fn.applyTransform(AlgebraSimplification.self) }
    case "LAF", "LinearAlgebraFusion":
        module.forEach { fn in fn.applyTransform(LinearAlgebraFusion.self) }
    case "SP", "StackPromotion":
        module.mapTransform(StackPromotion.self)
    case "VP", "ValuePromotion":
        module.mapTransform(ValuePromotion.self)
    case "MCO", "MatrixChainOrdering":
        module.forEach { fn in fn.mapTransform(MatrixChainOrdering.self) }
    case "LBP", "LiteralBroadcastingPromotion":
        module.forEach { fn in fn.mapTransform(LiteralBroadcastingPromotion.self) }
    default:
        error("No transform pass named \(name)")
    }
}

