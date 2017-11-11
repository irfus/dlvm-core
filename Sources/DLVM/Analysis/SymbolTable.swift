//
//  SymbolTable.swift
//  DLVM
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

open class SymbolTableAnalysis<Unit : IRCollection> : AnalysisPass
    where Unit.Iterator.Element : Named {
    public typealias Body = Unit
    public typealias Result = [String : Body.Iterator.Element]
    open class func run(on body: Body) -> Result {
        var table: Result = [:]
        for element in body {
            table[element.name] = element
        }
        return table
    }
}

public extension IRCollection where Iterator.Element : Named {
    func element(named name: String) -> Iterator.Element? {
        /// Guaranteed not to throw
        let table = analysis(from: SymbolTableAnalysis<Self>.self)
        return table[name]
    }
}
