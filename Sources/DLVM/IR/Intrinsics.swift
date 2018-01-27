//
//  Intrinsics.swift
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

/// An intrinsic function
public protocol IntrinsicProtocol {
    var opcode: String { get }
    func resultType(for operands: [Use]) -> Type
    func isEqualTo(_ other: IntrinsicProtocol) -> Bool
}

/// Dummy
public protocol BinaryIntrinsic : IntrinsicProtocol {}

public enum NumericBinaryIntrinsic : Equatable {
    case max, min
}

extension NumericBinaryIntrinsic : BinaryIntrinsic {
    public var opcode: String {
        switch self {
        case .max: return "max"
        case .min: return "min"
        }
    }

    public func resultType(for operands: [Use]) -> Type {
        guard operands.count == 2,
            case let .tensor(s1, dt1) = operands[0].type,
            case let .tensor(s2, dt2) = operands[1].type,
            let bcShape = s1.broadcast(with: s2), dt1 == dt2, dt1.isNumeric else {
                return .invalid
        }
        return .tensor(bcShape, dt1)
    }

    public func isEqualTo(_ other: IntrinsicProtocol) -> Bool {
        guard let op = (other as? NumericBinaryIntrinsic) else { return false }
        return self == op
    }
}
