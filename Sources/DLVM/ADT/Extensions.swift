//
//  Extensions.swift
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

public extension Sequence {
    /// Returns true if all elements satisfy the predicate
    func forAll(_ predicate: (Iterator.Element) -> Bool) -> Bool {
        return reduce(true, { $0 && predicate($1) })
    }

    /// Elements' descriptions joined by comma
    var joinedDescription: String {
        return map{"\($0)"}.joined(separator: ", ")
    }
    
    /// `mapM`
    func liftedMap(_ transform: (Iterator.Element) -> Iterator.Element?) -> [Iterator.Element]? {
        var result: [Iterator.Element] = []
        for x in self {
            guard let new = transform(x) else { return nil }
            result.append(new)
        }
        return result
    }
}

// MARK: - Equatable
public extension Sequence where Iterator.Element : Equatable {
    func except(_ exception: Iterator.Element) -> LazyFilterSequence<Self> {
        return lazy.filter { $0 != exception }
    }
}

// MARK: - Optional description
public extension Optional {
    var optionalDescription: String {
        return map{"\($0)"} ?? ""
    }
}

// MARK: - Description for TextOutputStreamable
internal extension TextOutputStreamable {
    var description: String {
        var desc = ""
        write(to: &desc)
        return desc
    }
}

/// Optional assignment
infix operator =?
public func =? <T> (lhs: inout T, rhs: T?) {
    if let rhs = rhs {
        lhs = rhs
    }
}
