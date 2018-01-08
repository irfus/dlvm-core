//
//  Location.swift
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

/// Two dimentional text location with line number, column number and linear index
public protocol TextLocation : Strideable {
    var line: Int { set get }
    var column: Int { set get }
    var index: Int { set get }
    static var initialPosition: Int { get }
    init(line: Int, column: Int, index: Int)
}

public extension TextLocation {
    init() {
        self.init(line: Self.initialPosition, column: Self.initialPosition, index: 0)
    }

    static func <(lhs: Self, rhs: Self) -> Bool {
        return lhs.index < rhs.index
    }

    static func ==(lhs: Self, rhs: Self) -> Bool {
        return lhs.column == rhs.column && lhs.line == rhs.line && lhs.index == rhs.index
    }

    func advanced(byLines lines: Int, columns: Int, distance: Int) -> Self {
        return Self(line: line + lines, column: column + columns, index: index + distance)
    }

    func advanced(by n: Int) -> Self {
        return Self(line: line, column: column + n, index: index + n)
    }

    func advanced(past character: UnicodeScalar) -> Self {
        return character == "\n" ? newLine() : advanced(by: 1)
    }

    static func + (lhs: Self, n: Int) -> Self {
        return lhs.advanced(by: n)
    }

    func advanced<S: Sequence>(byScanning prefix: S) -> Self
        where S.Element == UnicodeScalar
    {
        var new = self
        for char in prefix {
            if char == "\n" {
                new.line += 1
                new.column = Self.initialPosition
            } else {
                new.column += 1
            }
            new.index += 1
        }
        return new
    }

    func newLine() -> Self {
        return Self(line: line + 1, column: Self.initialPosition, index: index + 1)
    }

    mutating func advance(by n: Int) {
        column += n
        index += n
    }

    mutating func advanceToNewLine() {
        line += 1
        column = Self.initialPosition
        index += 1
    }

    func distance(to other: SourceLocation) -> Int {
        return other.index - index
    }

    var isBeginningOfLine: Bool {
        return column == Self.initialPosition
    }
}

/// Text location for source code
/// Initial position starts from 1
public struct SourceLocation : TextLocation {
    public typealias Stride = Int
    public static let initialPosition = 1
    public var line, column, index: Int

    public init(line: Int, column: Int, index: Int) {
        self.line = line
        self.column = column
        self.index = index
    }
}

extension SourceLocation : CustomStringConvertible {
    public var description: String {
        return "\(line):\(column)"
    }
}

public typealias SourceRange = CountableRange<SourceLocation>
