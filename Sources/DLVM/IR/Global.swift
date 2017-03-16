//
// Created by Richard Wei on 2/24/17.
//

/// TODO: Sum types for module's elements
public enum TopLevelItem {
    case value(GlobalValue)
    case function(Function)
    case `struct`(StructType)
}

/// Global value
public class GlobalValue : SimpleValue, Definition {
    public enum Kind {
        case variable, constant
    }
    public var name: String?
    public var kind: Kind
    public var shape: TensorShape
    public var dataType: DataType
    public var initializer: Literal

    public init(name: String?, kind: Kind, shape: TensorShape, dataType: DataType, initializer: Literal) {
        self.name = name
        self.kind = kind
        self.shape = shape
        self.dataType = dataType
        self.initializer = initializer
    }
}

/// Struct type
public struct StructType : Named {
    public var name: String
    public var fields: [Type]
}
