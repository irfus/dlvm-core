//
//  Gradient.swift
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

fileprivate struct GradientKey {
    let function: Function
    let differentiationIndex: Int
    let variableIndices: [Int]
}

extension GradientKey : Equatable, Hashable {
    static func == (lhs: GradientKey, rhs: GradientKey) -> Bool {
        return lhs.function === rhs.function
            && lhs.differentiationIndex == rhs.differentiationIndex
            && lhs.variableIndices == rhs.variableIndices
    }

    var hashValue: Int {
        return function.hashValue
    }
}

public struct GlobalGradientInfo {
    fileprivate var gradientMap: [GradientKey : Function] = [:]
    fileprivate var antigradientMap: [Function : Function] = [:]
}

public extension GlobalGradientInfo {
    func gradient(of function: Function,
                  from diffIndex: Int,
                  wrt varIndices: [Int]) -> Function? {
        let key = GradientKey(function: function,
                              differentiationIndex: diffIndex,
                              variableIndices: varIndices)
        return gradientMap[key]
    }

    func antigradient(of function: Function) -> Function? {
        return antigradientMap[function]
    }
}

public class GlobalGradientAnalysis: AnalysisPass<Module, GlobalGradientInfo> {
    public override class func run(on module: Module) -> GlobalGradientInfo {
        var ggi = GlobalGradientInfo()
        for grad in module {
            guard let key: GradientKey = grad.attributes.flatMap({ attr in
                guard case let .differentiating(f, from: diffIndex, wrt: varIndices) = attr
                    else { return nil }
                return GradientKey(function: f,
                                   differentiationIndex: diffIndex,
                                   variableIndices: varIndices)
            }).first else { continue }
            ggi.gradientMap[key] = grad
            ggi.antigradientMap[grad] = key.function
        }
        return ggi
    }
}
