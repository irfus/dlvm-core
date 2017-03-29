//
//  Extensions.swift
//  DLVM
//
//  Created by Richard Wei on 3/25/17.
//
//

public extension Sequence {
    func forAll(_ predicate: (Iterator.Element) -> Bool) -> Bool {
        return reduce(true, { $0 && predicate($1) })
    }

    var joinedDescription: String {
        return map{"\($0)"}.joined(separator: ", ")
    }
}

public extension Sequence where Iterator.Element : Equatable {
    public func except(_ exception: Iterator.Element) -> LazyFilterSequence<Self> {
        return lazy.filter { $0 != exception }
    }
}
