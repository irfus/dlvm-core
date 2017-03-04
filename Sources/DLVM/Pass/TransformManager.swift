//
//  TransformManager.swift
//  DLVM
//
//  Created by Richard Wei on 3/3/17.
//
//

open class TransformManager<Body : IRUnit> {
    public typealias TransformType = TransformPass<Body>
    fileprivate var passes: [TransformType] = []
}

internal extension TransformManager {
    
    func append(_ transform: TransformType) {
        passes.append(transform)
    }

    func append(_ transforms: TransformType...) {
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
    public func applyTransform<Transform>(_: Transform.Type) throws -> Bool
        where Transform : TransformManager<Self>.TransformType
    {
        let changed = try Transform.run(on: self)
        if Transform.shouldInvalidateAnalyses, changed {
            invalidateAnalyses()
        }
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
