//
//  TransformManager.swift
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

/*
/// Transform queue, a builder API for defining and applying a sequence
/// of transform passes
public class TransformQueue<Body : IRCollection> where Body.Iterator.Element : IRUnit {
    fileprivate enum Action {
        case apply((Body) throws -> Bool, shouldInvalidate: Bool)
        case map((Body.Iterator.Element) throws -> Bool, shouldInvalidate: Bool)
    }
    fileprivate var workList: [Action] = []
}

// MARK: - Transform queue builder
public extension TransformQueue {
    /// Build a 'map' action to the queue
    @discardableResult
    func map<T : TransformPass>(_: T.Type) -> TransformQueue where T.Body == Body.Element {
        workList.append(.map(T.run, shouldInvalidate: T.shouldInvalidateAnalyses))
        return self
    }

    /// Build an 'apply' action to the queue
    @discardableResult
    func apply<T : TransformPass>(_: T.Type) -> TransformQueue where T.Body == Body {
        workList.append(.apply(T.run, shouldInvalidate: T.shouldInvalidateAnalyses))
        return self
    }

    /// Apply queued transform passes to
    @discardableResult
    func run(on body: Body) throws -> Bool {
        var changed = false
        for functor in workList {
            switch functor {
            case let .apply(fn, shouldInvalidate: inv):
                changed = try fn(body) || changed
                if inv, changed {
                    body.invalidateAnalyses()
                }
            case let .map(fn, shouldInvalidate: inv):
                changed = try body.reduce(changed, { alreadyChanged, element in
                    let changed = try fn(element)
                    if inv, changed {
                        element.invalidateAnalyses()
                    }
                    return changed || alreadyChanged
                })
            }
        }
        return changed
    }
}
 */

public extension IRCollection {
    /// Applies a transform pass on self
    ///
    /// - Returns: whether changes are made
    @discardableResult
    func applyTransform<Transform : TransformPass>(_ transform: Transform.Type) throws -> Bool
        where Transform.Body == Self
    {
        let changed = try transform.run(on: self)
        if transform.shouldInvalidateAnalyses, changed {
            invalidateAnalyses()
        }
        /// Run verifier
        try verify()
        return changed
    }
}

public extension IRCollection {
    @discardableResult
    func mapTransform<Transform : TransformPass>(_ transform: Transform.Type) throws -> Bool
        where Transform.Body == Iterator.Element
    {
        var changed = false
        for element in self {
            changed = try element.applyTransform(transform) || changed
        }
        return changed
    }
}
