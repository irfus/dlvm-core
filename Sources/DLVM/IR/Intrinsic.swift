//
//  Intrinsic.swift
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
