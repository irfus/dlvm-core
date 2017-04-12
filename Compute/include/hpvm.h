//
//  hpvm.h
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

#import "swiftdecl.h" // Swift Clang-importer macros
#import <stdlib.h>

// Placeholder types
struct HPVM {};
struct HPVMAtomic {} SWIFT_NAME(HPVM.Atomic);
struct HPVMRuntime {} SWIFT_NAME(HPVM.Runtime);

typedef SWIFT_ENUM_NAMED(int, Target, "HPVM.Target") {
    TargetNone = 0,
    TargetCPU SWIFT_NAME(cpu),
    TargetGPU SWIFT_NAME(gpu),
    TargetSPIR SWIFT_NAME(spir),
    TargetCount
};

void __hpvm__hint(Target) SWIFT_NAME(HPVM.hint(_:));
void __hpvm__wait(int);

void __attribute__((noinline)) __hpvm__attributes(int, ...);
void __hpvm__init(); // Separately wrapped in hpvm.swift
void __hpvm__cleanup(); // Separately wrapped in hpvm.swift
void __hpvm__bindIn(void *_Nonnull, int, int, int) SWIFT_NAME(HPVM.bindIn(_:_:_:_:));
void __hpvm__bindOut(void *_Nonnull, int, int, int) SWIFT_NAME(HPVM.bindOut(_:_:_:_:));
void *_Nonnull __hpvm__edge(void *_Nonnull, void *_Nonnull, int, int, int, int) SWIFT_NAME(HPVM.bindOut(_:_:_:_:_:_:));
void __hpvm__push(void *_Nonnull, void *_Nonnull) SWIFT_NAME(HPVM.push(_:_:));
void *_Nonnull __hpvm__pop(void *_Nonnull) SWIFT_NAME(HPVM.pop(_:));
void *_Nonnull __attribute__((noinline)) __hpvm__launch(int, ...);
void *_Nonnull __hpvm__getNode(); // Separately wrapped in hpvm.swift
void *_Nonnull __hpvm__getParentNode(void *_Nonnull) SWIFT_NAME(HPVM.parentNode(of:));
void __hpvm__barrier(); // Separately wrapped in hpvm.swift
void *_Nonnull __hpvm__malloc(long) SWIFT_NAME(HPVM.malloc(_:));
int __hpvm__getNodeInstanceID_x(void *_Nonnull) SWIFT_NAME(HPVM.nodeInstanceIdX(of:));
int __hpvm__getNodeInstanceID_y(void *_Nonnull) SWIFT_NAME(HPVM.nodeInstanceIdY(of:));
int __hpvm__getNodeInstanceID_z(void *_Nonnull) SWIFT_NAME(HPVM.nodeInstanceIdZ(of:));
int __hpvm__getNumNodeInstances_x(void *_Nonnull) SWIFT_NAME(HPVM.nodeInstanceCountX(of:));
int __hpvm__getNumNodeInstances_y(void *_Nonnull) SWIFT_NAME(HPVM.nodeInstanceCountY(of:));
int __hpvm__getNumNodeInstances_z(void *_Nonnull) SWIFT_NAME(HPVM.nodeInstanceCountZ(of:));

// Atomic
int __hpvm__atomic_cmpxchg(int *_Nonnull, int, int) SWIFT_NAME(HPVMAtomic.cmpxchg(_:_:_:));
int __hpvm__atomic_add(int *_Nonnull, int) SWIFT_NAME(HPVMAtomic.add(_:_:));
int __hpvm__atomic_sub(int *_Nonnull, int) SWIFT_NAME(HPVMAtomic.sub(_:_:));
int __hpvm__atomic_xchg(int *_Nonnull, int) SWIFT_NAME(HPVMAtomic.xchg(_:_:));
int __hpvm__atomic_inc(int *_Nonnull) SWIFT_NAME(HPVMAtomic.inc(_:));
int __hpvm__atomic_dec(int *_Nonnull) SWIFT_NAME(HPVMAtomic.dec(_:));
int __hpvm__atomic_min(int *_Nonnull, int) SWIFT_NAME(HPVMAtomic.min(_:_:));
int __hpvm__atomic_max(int *_Nonnull, int) SWIFT_NAME(HPVMAtomic.max(_:_:));
int __hpvm__atomic_umax(int *_Nonnull, int) SWIFT_NAME(HPVMAtomic.umax(_:_:));
int __hpvm__atomic_umin(int *_Nonnull, int) SWIFT_NAME(HPVMAtomic.umin(_:_:));
int __hpvm__atomic_and(int *_Nonnull, int) SWIFT_NAME(HPVMAtomic.and(_:_:));
int __hpvm__atomic_or(int *_Nonnull, int) SWIFT_NAME(HPVMAtomic.or(_:_:));
int __hpvm__atomic_xor(int *_Nonnull, int) SWIFT_NAME(HPVMAtomic.xor(_:_:));

// Special functions
float __hpvm__floor(float) SWIFT_NAME(HPVM.floor(_:));
float __hpvm__rsqrt(float) SWIFT_NAME(HPVM.rsqrt(_:));
float __hpvm__sqrt(float) SWIFT_NAME(HPVM.sqrt(_:));
float __hpvm__sin(float) SWIFT_NAME(HPVM.sin(_:));

void llvm_hpvm_track_mem(void *_Nonnull, size_t) SWIFT_NAME(HPVMRuntime.trackMemory(_:ofSize:));
void llvm_hpvm_untrack_mem(void *_Nonnull) SWIFT_NAME(HPVMRuntime.untrackMemory(_:));
void llvm_hpvm_request_mem(void *_Nonnull, size_t) SWIFT_NAME(HPVMRuntime.requestMemory(_:ofSize:));


/// Specialize and overload variadics since Swift Clang importer does not natively support variadics

inline void __hpvm__attributes_3_1(int, const void *_Nonnull, const void *_Nonnull, const void *_Nonnull, int, const void *_Nonnull)
    SWIFT_NAME(HPVM.attributes(_:_:_:_:_:_:));

inline void __hpvm__attributes_3_1(int a1, const void *_Nonnull a2, const void *_Nonnull a3, const void *_Nonnull a4,
                                   int a5, const void *_Nonnull a6) {
    __hpvm__attributes(a1, a2, a3, a4, a5, a6);
}

inline void __hpvm__launch_1(int) SWIFT_NAME(HPVM.launch(_:));

inline void __hpvm__launch_1(int a1) {
    __hpvm__launch(a1);
}
