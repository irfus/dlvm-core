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

public enum Target : Int {
    case none = 0
    case cpu = 1
    case gpu = 2
    case spir = 3
    case targetCount = 4
}

public typealias P<T> = UnsafeMutablePointer<T>
public typealias P_ = UnsafeMutableRawPointer
public typealias I8 = CChar
public typealias U8 = CUnsignedChar

public enum HPVM {
    public enum Atomic {}
}

public extension HPVM {

    @_silgen_name("llvm.hpvm.hint")
    static func hint(_: Target)

    @_silgen_name("llvm.hpvm.wait")
    static func wait(_: U8)

    @_silgen_name("llvm.hpvm.attributes")
    static func attributes(_: P<U8>)

    @_silgen_name("llvm.hpvm.init")
    static func `init`()

    @_silgen_name("llvm.hpvm.cleanup")
    static func cleanup()

    @_silgen_name("llvm.hpvm.bindIn")
    static func bindIn(_: P_, _: U8, _: U8, _: U8)

    @_silgen_name("llvm.hpvm.bindOut")
    static func bindOut(_: P_, _: U8, _: U8, _: U8)

    @_silgen_name("llvm.hpvm.edge")
    static func edge(_: P_, _: P_, _: U8, _: U8, _: U8, _: U8) -> P_

    @_silgen_name("llvm.hpvm.push")
    static func push(_: P_, _: P_)

    @_silgen_name("llvm.hpvm.pop")
    static func pop(_: P_) -> P_

    @_silgen_name("llvm.hpvm.launch")
    static func launch(_: P_) -> P_

    @_silgen_name("llvm.hpvm.getNode")
    static func getNode() -> P_

    @_silgen_name("llvm.hpvm.getParentNode")
    static func getParentNode(_: P_) -> P_

    @_silgen_name("llvm.hpvm.barrier")
    static func barrier()

    @_silgen_name("llvm.hpvm.malloc")
    static func malloc(_: CUnsignedLong) -> P_

    @_silgen_name("llvm.hpvm.getNodeInstanceID.x")
    static func getNodeInstanceIdX(_: P_) -> U8

    @_silgen_name("llvm.hpvm.getNodeInstanceID.y")
    static func getNodeInstanceIdY(_: P_) -> U8

    @_silgen_name("llvm.hpvm.getNodeInstanceID.z")
    static func getNodeInstanceIdZ(_: P_) -> U8

    @_silgen_name("llvm.hpvm.getNumNodeInstances_x(void*)")
    static func getNodeInstanceCountX(_: P_) -> U8

    @_silgen_name("llvm.hpvm.getNumNodeInstances_y(void*)")
    static func getNodeInstanceCountY(_: P_) -> U8

    @_silgen_name("llvm.hpvm.getNumNodeInstances_z(void*)")
    static func getNodeInstanceCountZ(_: P_) -> U8

    @_silgen_name("llvm_hpvm_track_mem")
    static func trackMemory(_: P_, _: CUnsignedLong)

    @_silgen_name("llvm_hpvm_untrack_mem")
    static func untrackMemory(_: P_)

    @_silgen_name("llvm_hpvm_request_mem")
    static func requestMemory(_: P_, _: CUnsignedLong)

    /// Spcial functions

    @_silgen_name("llvm.hpvm.floor")
    static func floor(_: CFloat) -> CFloat

    @_silgen_name("llvm.hpvm.rsqrt")
    static func rsqrt(_: CFloat) -> CFloat

    @_silgen_name("llvm.hpvm.sqrt")
    static func sqrt(_: CFloat) -> CFloat

    @_silgen_name("llvm.hpvm.sin")
    static func sin(_: CFloat) -> CFloat

}

extension HPVM.Atomic {

    @_silgen_name("llvm.hpvm.atomic.max")
    func cmpxchg(_: P<CInt>, _: CInt, _: CInt);

    @_silgen_name("llvm.hpvm.atomic.add")
    func add(_: P<CInt>, _: CInt);

    @_silgen_name("llvm.hpvm.atomic.sub")
    func sub(_: P<CInt>, _: CInt);

    @_silgen_name("llvm.hpvm.atomic.xchg")
    func xchg(_: P<CInt>, _: CInt);

    @_silgen_name("llvm.hpvm.atomic.inc")
    func inc(_: P<CInt>);

    @_silgen_name("llvm.hpvm.atomic.dec")
    func dec(_: P<CInt>);

    @_silgen_name("llvm.hpvm.atomic.min")
    func min(_: P<CInt>, _: CInt);

    @_silgen_name("llvm.hpvm.atomic.max")
    func max(_: P<CInt>, _: CInt);

    @_silgen_name("llvm.hpvm.atomic.umax")
    func umax(_: P<CInt>, _: CInt);

    @_silgen_name("llvm.hpvm.atomic.umin")
    func umin(_: P<CInt>, _: CInt);

    @_silgen_name("llvm.hpvm.atomic.and")
    func and(_: P<CInt>, _: CInt);

    @_silgen_name("llvm.hpvm.atomic.or")
    func or(_: P<CInt>, _: CInt);

    @_silgen_name("llvm.hpvm.atomic.xor")
    func xor(_: P<CInt>, _: CInt);

}
