//
// Created by Richard Wei on 2/24/17.
//

public struct Use {

    public enum Kind {
        case argument(Def<Argument>)
        case local(Def<Operation>)
        case global(Def<GlobalValue>)
        case literal(LiteralValue)
    }

    public var shape: TensorShape
    public var type: DataType
    public var kind: Kind

    public init(shape: TensorShape, type: DataType, kind: Kind) {
        self.shape = shape
        self.type = type
        self.kind = kind
    }

    public init(kind: Kind) {
        switch kind {
        case .local(let def):
            self.init(shape: def.shape, type: def.type, kind: kind)
        case .global(let def):
            self.init(shape: def.shape, type: def.type, kind: kind)
        case .argument(let def):
            self.init(shape: def.shape, type: def.type, kind: kind)
        case .literal(let lit):
            self.init(shape: lit.shape, type: lit.type, kind: kind)
        }
    }

}

// MARK: - Equatable
extension Use : Equatable {
    public static func ==(lhs: Use, rhs: Use) -> Bool {
        return lhs.definition === rhs.definition
               && lhs.name == rhs.name
               && lhs.shape == rhs.shape
               && lhs.type == rhs.type
    }
}

// MARK: - Factory methods
public extension Use {

    static func local(_ definition: Def<Operation>) -> Use {
        return Use(kind: .local(definition))
    }

    static func global(_ definition: Def<GlobalValue>) -> Use {
        return Use(kind: .global(definition))
    }

    static func argument(_ definition: Def<Argument>) -> Use {
        return Use(kind: .argument(definition))
    }

    static func literal(_ literal: Literal, shape: TensorShape, type: DataType) -> Use {
        return Use(kind: .literal(LiteralValue(shape: shape, type: type, literal: literal)))
    }

    static func literal(_ literalValue: LiteralValue) -> Use {
        return Use(kind: .literal(literalValue))
    }

}

// MARK: - Value properties
public extension Use {

    var definition: AnyDef? {
        switch kind {
        case let .global(def): return def
        case let .local(def): return def
        case let .argument(def): return def
        case .literal: return nil
        }
    }

    var value: Value {
        switch kind {
        case let .global(def): return def
        case let .local(def): return def
        case let .argument(def): return def
        case let .literal(lit): return lit
        }
    }

    var name: String? {
        switch kind {
        case let .global(def): return def.name
        case let .local(def): return def.name
        case let .argument(def): return def.name
        case .literal: return nil
        }
    }
}

// MARK: - Value helper factories
public extension Use {
    func makeZero() -> LiteralValue {
        return value.makeZero()
    }
}
