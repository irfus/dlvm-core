//
//  Intrinsic.swift
//  DLVM
//
//  Created by Richard Wei on 2/21/17.
//
//

/*

public struct Intrinsic {
    public var name: String
    public var kind: OpKind

    public typealias TypeSignature = (arguments: [DataType], result: DataType)
    /// Type signature is either fully specified, or generically type-monomorphic (when nil)
    public var typeSignature: TypeSignature?
}

public extension Intrinsic {
    /// Returns the result shape for arguments,
    /// or nil if arguments are invalid
    ///
    /// - Parameter args: arguments
    func resultShape(forArguments args: [TensorShape]) -> TensorShape? {
        return kind.resultShape(forArguments: args)
    }

    /// Returns the result type for arguments,
    /// or nil if arguments are invalid
    ///
    /// - Parameter args: arguments
    func resultType(forArguments args: [DataType]) -> DataType? {
        guard let first = args.first else { return nil }
        guard let signature = typeSignature else { return first }
        guard args == signature.arguments else { return nil }
        return signature.result
    }

    /// Returns the result type and shape for arguments,
    /// or nil if arguments are invalid
    ///
    /// - Parameter args: arguments
    func result(forArguments args: [Use]) -> (TensorShape, DataType)? {
        return resultShape(forArguments: args.map{$0.shape}).flatMap { shape in
            resultType(forArguments: args.map{$0.type}).flatMap { type in
                (shape, type)
            }
        }
    }
}

 */
