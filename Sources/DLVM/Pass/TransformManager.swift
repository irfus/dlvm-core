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

public class TransformManager<Body : IRUnit> {
    public typealias TransformType = TransformPass<Body>
    public internal(set) var performedTransforms: [TransformType.Type] = []
}

// MARK: - Mutation
internal extension TransformManager {
    func append(_ transform: TransformType.Type) {
        performedTransforms.append(transform)
    }

    func append(_ transforms: TransformType.Type...) {
        for transform in transforms {
            append(transform)
        }
    }
}

/// - Note: Currently TransformManager is not being utilized,
/// transforms are run from IRUnit directly
public extension IRUnit {
    /// Applies a transform pass on self
    ///
    /// - Returns: whether changes are made
    @discardableResult
    func applyTransform<Transform>(_ transform: Transform.Type) throws -> Bool
        where Transform : TransformManager<Self>.TransformType
    {
        let changed = try transform.run(on: self)
        transformManager.append(transform)
        if transform.shouldInvalidateAnalyses, changed {
            invalidateAnalyses()
        }
        /// Run verifier
        try verify()
        return changed
    }

    /// Applies transform passes on self
    ///
    /// - Returns: whether changes are made
    @discardableResult
    func applyTransforms<Transform>(_ transforms: Transform.Type...) throws -> Bool
        where Transform : TransformManager<Self>.TransformType
    {
        var changed = false
        for transform in transforms {
            changed = try applyTransform(transform)
        }
        return changed
    }
}
