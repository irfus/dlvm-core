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

// MARK: - Recurrent value helper
public extension Global {
    var isRecurrent: Bool {
        switch self {
        case let .placeholder(def):
            return def.value.isRecurrent
        case let .output(def):
            return def.value.isRecurrent
        default:
            return false
        }
    }
}

/// Global value
public struct GlobalValue : Value {
    public enum Kind {
        case variable, constant
    }
    public var kind: Kind
    public var shape: TensorShape
    public var type: DataType
    public var initializer: Literal
    public static let scope: Scope = .global

    public init(kind: Kind, shape: TensorShape, type: DataType, initializer: Literal) {
        self.kind = kind
        self.shape = shape
        self.type = type
        self.initializer = initializer
    }
}

/// Placeholder in a DFG
/// - Note: Although it goes into a `Def`, it's never `Use`able!
public struct Placeholder : PotentiallyRecurrentValue {
    public var shape: TensorShape
    public var type: DataType
    public var isRecurrent: Bool
    public static var scope: Scope = .global

    public init(shape: TensorShape, type: DataType, isRecurrent: Bool) {
        self.shape = shape
        self.type = type
        self.isRecurrent = isRecurrent
    }
}

/// Output of a DFG
/// - Note: Although it goes into a `Def`, it's never `Use`able!
public struct Output : PotentiallyRecurrentValue {
    public var shape: TensorShape
    public var type: DataType
    public var isRecurrent: Bool
    public static var scope: Scope = .global

    public init(shape: TensorShape, type: DataType, isRecurrent: Bool) {
        self.shape = shape
        self.type = type
        self.isRecurrent = isRecurrent
    }
}
