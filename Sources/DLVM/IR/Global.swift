//
// Created by Richard Wei on 2/24/17.
//

/// Global definition, either `Def<GlobalValue>`, `Def<Placeholder>`
/// or `Def<Output>`
public enum Global {
    case value(Def<GlobalValue>)
    case placeholder(Def<Placeholder>)
    case output(Def<Output>)
}

/// Global value
public struct GlobalValue : SimpleValue {
    public enum Kind {
        case variable, constant
    }
    public var kind: Kind
    public var shape: TensorShape
    public var dataType: DataType
    public var initializer: Literal
    public static let scope: Scope = .global

    public init(kind: Kind, shape: TensorShape, type: DataType, initializer: Literal) {
        self.kind = kind
        self.shape = shape
        self.dataType = type
        self.initializer = initializer
    }
}

/// Placeholder in a DFG
/// - Note: Although it goes into a `Def`, it's never `Use`able!
public struct Placeholder : PotentiallyRecurrentValue, SimpleValue {
    public var shape: TensorShape
    public var dataType: DataType
    public var isRecurrent: Bool
    public static var scope: Scope = .global

    public init(shape: TensorShape, type: DataType, isRecurrent: Bool) {
        self.shape = shape
        self.dataType = type
        self.isRecurrent = isRecurrent
    }
}

/// Output of a DFG
/// - Note: Although it goes into a `Def`, it's never `Use`able!
public struct Output : PotentiallyRecurrentValue, SimpleValue {
    public var shape: TensorShape
    public var dataType: DataType
    public var isRecurrent: Bool
    public static var scope: Scope = .global

    public init(shape: TensorShape, type: DataType, isRecurrent: Bool) {
        self.shape = shape
        self.dataType = type
        self.isRecurrent = isRecurrent
    }
}
