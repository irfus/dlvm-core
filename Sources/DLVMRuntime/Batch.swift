//
//  Batch.swift
//  DLVM
//
//  Created by Richard Wei on 2/5/17.
//
//

import DLVM
import Foundation
import dlrt

public protocol BatchProtocol : RandomAccessCollection {
    associatedtype Element
}

public protocol BatchStream {
    associatedtype ItemType
}

open class Batch<ItemType, Source : BatchStream> : BatchProtocol
    where Source.ItemType == ItemType {

    internal var data: Tensor<ItemType>

    open internal(set) var size: Int

    open var source: Source?

    public init(data: Tensor<ItemType>) {
        self.data = data
        self.size = data.count
    }

    public init(size: Int, source: Source, elementShape: TensorShape) {
        self.size = size
        self.source = source
        data = Tensor(elementShape: elementShape)
    }

}

extension Batch : RandomAccessCollection {

    public typealias Index = Int
    public typealias IndexDistance = Int
    public typealias Element = Tensor<ItemType>

    open var count: Int {
        return data.count
    }

    open var indices: CountableRange<Int> {
        return data.indices
    }

    open subscript(i: Int) -> Tensor<ItemType> {
        get {
            return data[i]
        }
        set {
            data[i] = newValue
        }
    }

    open func index(after i: Int) -> Int {
        return i + 1
    }

    open func index(before i: Int) -> Int {
        return i - 1
    }

    open var startIndex: Int {
        return 0
    }

    open var endIndex: Int {
        return count
    }

}
