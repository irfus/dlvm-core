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
        try module.applyTransform(Differentiation.self)
    case "Can", "Canonicalization":
        try module.applyTransform(Canonicalization.self)
    case "CP", "Checkpointing":
        try module.mapTransform(Checkpointing.self)
    case "DCE", "DeadCodeElimination":
        try module.mapTransform(DeadCodeElimination.self)
    case "CSE", "CommonSubexpressionElimination":
        try module.mapTransform(CommonSubexpressionElimination.self)
    case "AS", "AlgebraSimplification":
        try module.forEach { fn in try fn.applyTransform(AlgebraSimplification.self) }
    case "LAF", "LinearAlgebraFusion":
        try module.forEach { fn in try fn.applyTransform(LinearAlgebraFusion.self) }
    case "SP", "StackPromotion":
        try module.mapTransform(StackPromotion.self)
    case "VP", "ValuePromotion":
        try module.mapTransform(ValuePromotion.self)
    case "MCO", "MatrixChainOrdering":
        try module.forEach { fn in try fn.mapTransform(MatrixChainOrdering.self) }
    default:
        error("No transform pass named \(name)")
    }
}

