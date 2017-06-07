//
//  SymbolTable.swift
//  DLVM
//
//  Created by Richard Wei on 6/7/17.
//
//

open class SymbolTableAnalysis<Unit : IRCollection> : AnalysisPass
    where Unit.Iterator.Element : Named {
    public typealias Body = Unit
    public typealias Result = [String : Body.Iterator.Element]
    open class func run(on body: Body) throws -> Result {
        var table: Result = [:]
        for element in body {
            table[element.name] = element
        }
        return table
    }
}
