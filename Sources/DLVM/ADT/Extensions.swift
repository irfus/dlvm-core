//
//  Extensions.swift
//  DLVM
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

public extension Sequence {
    /// Returns true if all elements satisfy the predicate
    func forAll(_ predicate: (Element) throws -> Bool) rethrows -> Bool {
        return try first(where: { try !predicate($0) }) == nil
    }

    /// Elements' descriptions joined by comma
    var joinedDescription: String {
        return description(joinedBy: ", ")
    }

    /// Elements' descriptions joined
    func description(joinedBy separator: String) -> String {
        return map{"\($0)"}.joined(separator: separator)
    }
    
    /// `mapM`
    func liftedMap<Result>(_ transform: (Element) -> Result?) -> [Result]? {
        var result: [Result] = []
        for x in self {
            guard let new = transform(x) else { return nil }
            result.append(new)
        }
        return result
    }
}

func ==<A: Equatable, B: Equatable>(lhs: [(A, B)], rhs: [(A, B)]) -> Bool {
    return lhs.count == rhs.count && zip(lhs, rhs).forAll(==)
}

public extension Optional {
    @discardableResult
    func ifAny<T>(_ execute: (Wrapped) throws -> T) rethrows -> T? {
        return try flatMap(execute)
    }
}

public extension Collection where Index == Int {
    func subcollection(atIndices indices: [Int]) -> [Element]? {
        guard indices.count <= count else { return nil }
        var result: [Element] = []
        for index in indices {
            if index > count { return nil }
            result.append(self[index])
        }
        return result
    }
}

public extension Sequence where Element : Hashable {
    var containsDuplicate: Bool {
        var set: Set<Element> = []
        for element in self {
            if set.contains(element) { return true }
            set.insert(element)
        }
        return false
    }
}

public extension Sequence where Element : Equatable {
    func except(_ exception: Element) -> LazyFilterSequence<Self> {
        return lazy.filter { $0 != exception }
    }
}

public extension Optional {
    var optionalDescription: String {
        return map{"\($0)"} ?? ""
    }
}

/// Description for TextOutputStreamable
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

import struct Foundation.URL

public extension String {
    func replacingFileExtension(with newExtenion: String) -> String {
        let url = URL(fileURLWithPath: self)
        let dlUrl = url.deletingPathExtension().appendingPathExtension(newExtenion)
        return dlUrl.relativePath
    }
}
