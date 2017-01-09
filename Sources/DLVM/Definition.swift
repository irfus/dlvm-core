//
//  Definition.swift
//  DLVM
//
//  Created by Richard Wei on 1/7/17.
//
//

public protocol TensorDefinition : VariableProducer {
    var dataType: DataType { get }
    var shape: TensorShape { get }
}

public protocol ScalarDefinition : VariableProducer {
    var type: ScalarType { get }
}

public extension TensorDefinition {
    public func makeVariable(named name: String) -> VariableOperand? {
        return TensorVariable(name: name, dataType: dataType,
                              shape: shape, definition: self)
    }
}

public extension ScalarDefinition {
    public func makeVariable(named name: String) -> VariableOperand? {
        return ScalarVariable(name: name, type: type, definition: self)
    }
}

public class ImmediateTensorDefinition : TensorDefinition {
    public var dataType: DataType
    public var shape: TensorShape
    public var value: ImmediateOperand

    public init(dataType: DataType, shape: TensorShape, value: ImmediateOperand) {
        precondition(dataType ~= value.type)
        self.dataType = dataType
        self.shape = shape
        self.value = value
    }
}

public class RandomizingTensorDefinition : TensorDefinition {
    public var dataType: DataType
    public var shape: TensorShape
    public var lowerBound: ImmediateOperand
    public var upperBound: ImmediateOperand

    public init(dataType: DataType, shape: TensorShape,
                lowerBound: ImmediateOperand, upperBound: ImmediateOperand) {
        precondition(dataType ~= lowerBound.type && dataType ~= upperBound.type)
        self.dataType = dataType
        self.shape = shape
        self.lowerBound = lowerBound
        self.upperBound = upperBound
    }
}

public class ImmediateScalarDefinition : ScalarDefinition {
    public var type: ScalarType
    public var value: ImmediateOperand

    public init?(type: ScalarType, value: ImmediateOperand) {
        guard type == value.type else { return nil }
        self.type = type
        self.value = value
    }
}
