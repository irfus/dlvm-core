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

fileprivate struct GradientConfig {
    let function: Function
    let differentiationIndex: Int
    let variableIndices: [Int]
    let outputIndices: [Int]
    let isSeedable: Bool
}

extension GradientConfig : Equatable, Hashable {
    static func == (lhs: GradientConfig, rhs: GradientConfig) -> Bool {
        return lhs.function === rhs.function
            && lhs.differentiationIndex == rhs.differentiationIndex
            && lhs.variableIndices == rhs.variableIndices
            && lhs.outputIndices == rhs.outputIndices
            && lhs.isSeedable == rhs.isSeedable
    }

    var hashValue: Int {
        return function.hashValue
    }
}

public struct GradientRelationInfo {
    fileprivate var gradientMap: [GradientConfig : Function] = [:]
    fileprivate var antigradientMap: [Function : Function] = [:]
}

public extension GradientRelationInfo {
    func gradient(of function: Function,
                  from diffIndex: Int,
                  wrt varIndices: [Int],
                  keeping outputIndices: [Int],
                  isSeedable: Bool) -> Function? {
        let key = GradientConfig(function: function,
                              differentiationIndex: diffIndex,
                              variableIndices: varIndices,
                              outputIndices: outputIndices,
                              isSeedable: isSeedable)
        return gradientMap[key]
    }

    func antigradient(of function: Function) -> Function? {
        return antigradientMap[function]
    }
}

open class GradientRelationAnalysis: AnalysisPass {
    public typealias Body = Module
    public typealias Result = GradientRelationInfo
    
    open class func run(on module: Module) -> GradientRelationInfo {
        var ggi = GradientRelationInfo()
        for grad in module {
            guard let key: GradientConfig = grad.attributes.flatMap({ attr in
                guard case let .gradient(.function(_, f), from: diffIndex,
                                         wrt: varIndices,
                                         keeping: outputIndices,
                                         seedable: isSeedable) = attr
                    else { return nil }
                return GradientConfig(function: f,
                                      differentiationIndex: diffIndex,
                                      variableIndices: varIndices,
                                      outputIndices: outputIndices,
                                      isSeedable: isSeedable)
            }).first else { continue }
            ggi.gradientMap[key] = grad
            ggi.antigradientMap[grad] = key.function
        }
        return ggi
    }
}
