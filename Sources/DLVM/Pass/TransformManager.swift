//
//  TransformManager.swift
//  DLVM
//
//  Created by Richard Wei on 3/3/17.
//
//

open class TransformManager<Body : IRUnit> {
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

    /// Apply a transform pass on self
    ///
    /// - Returns: whether changes were made
    @discardableResult
    public func applyTransform<Transform>(_ transform: Transform.Type) throws -> Bool
        where Transform : TransformManager<Self>.TransformType
    {
        let changed = try transform.run(on: self)
        transformManager.append(transform)
        if transform.shouldInvalidateAnalyses, changed {
            invalidateAnalyses()
        }
        _ = try analysis(from: Verifier<Self>.self)
        return changed
    }

    @discardableResult
    public func applyTransforms<Transform>(_ transforms: Transform.Type...) throws -> Bool
        where Transform : TransformManager<Self>.TransformType
    {
        var changed = false
        for transform in transforms {
            changed = try applyTransform(transform)
        }
        return changed
    }
    
}
