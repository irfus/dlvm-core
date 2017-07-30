//
//  LinearAlgebraFusion.swift
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

/// Linear Algebra Fusion converts polynomials with matrix multiplication
/// terms to a single matrix multiplication applied to tensors concatenated
/// beforehand.

open class LinearAlgebraFusion : TransformPass {
    public typealias Body = Function

    private static func performFusion() {
        DLUnimplemented()
    }
    
    open class func run(on body: Function) -> Bool {
        DLUnimplemented()
    }
}
