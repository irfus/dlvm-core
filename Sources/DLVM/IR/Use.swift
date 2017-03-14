//
// Created by Richard Wei on 2/24/17.
//

public struct Use {
    public enum Kind {
        case argument(Argument)
        case local(Instruction)
        case global(GlobalValue)
        case literal(LiteralValue)
    }

    public var type: Type
    public var kind: Kind

    public init(type: Type, kind: Kind) {
        self.type = type
        self.kind = kind
    }

    public init(kind: Kind) {
        switch kind {
        case .local(let def):
            self.init(type: def.type, kind: kind)
        case .global(let def):
            self.init(type: def.type, kind: kind)
        case .argument(let def):
            self.init(type: def.type, kind: kind)
        case .literal(let lit):
            self.init(type: lit.type, kind: kind)
        }
    }
}

// MARK: - Equatable
extension Use : Equatable {
    public static func ==(lhs: Use, rhs: Use) -> Bool {
        return lhs.definition === rhs.definition
               && lhs.name == rhs.name
               && lhs.type == rhs.type
    }
}

// MARK: - Factory methods
public extension Use {

    static func local(_ definition: Instruction) -> Use {
        return Use(kind: .local(definition))
    }

    static func global(_ definition: GlobalValue) -> Use {
        return Use(kind: .global(definition))
    }

    static func argument(_ definition: Argument) -> Use {
        return Use(kind: .argument(definition))
    }

    static func literal(_ literal: Literal, shape: TensorShape, type: DataType) -> Use {
        return Use(kind: .literal(LiteralValue(shape: shape, dataType: type, literal: literal)))
    }

    static func literal(_ literalValue: LiteralValue) -> Use {
        return Use(kind: .literal(literalValue))
    }

}

// MARK: - Value properties
public extension Use {

    var value: Value {
        switch kind {
        case let .argument(val): return val
        case let .global(val): return val
        case let .literal(val): return val
        case let .local(val): return val
        }
    }

    var definition: Definition? {
        switch kind {
        case let .global(def): return def
        case let .argument(def): return def
        case .literal, .local: return nil
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
