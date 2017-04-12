//
//  hpvm.swift
//  DLVM Comptue Primitives
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

/// This file contains HPVM intrinsics mapped onto Swift functions

public typealias P<T> = UnsafeMutablePointer<T>
public typealias P_ = UnsafeMutableRawPointer
public typealias I32 = CInt

/// This extension contains wrappers of top-level void-argument
/// functions that cannot be imported using `SWIFT_NAME`
public extension HPVM {
    @_silgen_name("__hpvm__init")
    static func initialize()

    @_silgen_name("__hpvm__barrier")
    static func barrier()

    @_silgen_name("__hpvm__getNode")
    static func currentNode() -> P_

    @_silgen_name("__hpvm__cleanup")
    static func cleanup()
}
