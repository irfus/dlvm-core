//
//  IRCollection.swift
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

/// IRCollection
public protocol IRCollection : class, RandomAccessCollection, Verifiable, AnalysisCacheable {
    associatedtype Base : OrderedSetCollection
    associatedtype Element : Hashable = Base.Element
    associatedtype Index = Base.Index
    associatedtype IndexDistance = Base.IndexDistance
    associatedtype Iterator = Base.Iterator
    associatedtype Indices = Base.Indices
    associatedtype SubSequence : Collection = Base.SubSequence
    var elements: Base { get set }
    var canApplyTransforms: Bool { get }
}

public extension IRCollection
    where Base.Index == Index, Base.Iterator == Iterator, Base.Iterator.Element == Element,
          Base.Element == Element, Base.SubSequence == SubSequence, Base.Indices == Indices,
          Base.IndexDistance == IndexDistance {

    func makeIterator() -> Iterator {
        return elements.makeIterator()
    }

    func index(after i: Base.Index) -> Base.Index {
        return elements.index(after: i)
    }

    func index(before i: Base.Index) -> Base.Index {
        return elements.index(before: i)
    }

    var indices: Base.Indices {
        return elements.indices
    }

    var startIndex: Base.Index {
        return elements.startIndex
    }

    var endIndex: Base.Index {
        return elements.endIndex
    }

    subscript(i: Base.Index) -> Base.Element {
        return elements[i]
    }

    subscript(bounds: Range<Base.Index>) -> Base.SubSequence {
        return elements[bounds]
    }
}

public extension IRCollection
    where Base.Index == Index, Base.Element == Base.Iterator.Element, Base.Element == Element,
          Base.SubSequence == SubSequence, Base.Indices == Indices, Base.IndexDistance == IndexDistance,
          Element : IRUnit, Element.Parent == Self {

    func remove(_ element: Element) {
        elements.remove(element)
        invalidateAnalyses()
    }

    func contains(_ element: Element) -> Bool {
        return elements.contains(element)
    }

    func append(_ newElement: Element) {
        elements.append(newElement)
        newElement.parent = self
        invalidateAnalyses()
    }

    func insert(_ newElement: Element, at index: Base.Index) {
        elements.insert(newElement, at: index)
        newElement.parent = self
        invalidateAnalyses()
    }

    func insert(_ newElement: Element, after other: Element) {
        elements.insert(newElement, after: other)
        newElement.parent = self
        invalidateAnalyses()
    }

    func insert(_ newElement: Element, before other: Element) {
        elements.insert(newElement, before: other)
        newElement.parent = self
        invalidateAnalyses()
    }

}
