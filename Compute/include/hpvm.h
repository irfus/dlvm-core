/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#ifndef DEVICE
#define DEVICE GPU_TARGET
#endif

#import <stdlib.h>

enum Target {
    None,
    CPU_TARGET,
    GPU_TARGET,
    SPIR_TARGET,
    NUM_TARGETS
};

void __hpvm__hint(enum Target);
void __hpvm__wait(unsigned);
void __hpvm__attributes(unsigned, ...);
void __hpvm__init();
void __hpvm__cleanup();
void __hpvm__bindIn(void*, unsigned, unsigned, unsigned);
void __hpvm__bindOut(void*, unsigned, unsigned, unsigned);
void* __hpvm__edge(void*, void*, unsigned, unsigned, unsigned, unsigned);
void __hpvm__push(void*, void*);
void* __hpvm__pop(void*);
void* __hpvm__launch(unsigned, ...);
void* __hpvm__getNode();
void* __hpvm__getParentNode(void*);
void __hpvm__barrier();
void* __hpvm__malloc(long);
unsigned __hpvm__getNodeInstanceID_x(void*);
unsigned __hpvm__getNodeInstanceID_y(void*);
unsigned __hpvm__getNodeInstanceID_z(void*);
unsigned __hpvm__getNumNodeInstances_x(void*);
unsigned __hpvm__getNumNodeInstances_y(void*);
unsigned __hpvm__getNumNodeInstances_z(void*);

// Atomic
// signed int
int __hpvm__atomic_cmpxchg(int*, int, int);
int __hpvm__atomic_add(int*, int);
int __hpvm__atomic_sub(int*, int);
int __hpvm__atomic_xchg(int*, int);
int __hpvm__atomic_inc(int*);
int __hpvm__atomic_dec(int*);
int __hpvm__atomic_min(int*, int);
int __hpvm__atomic_max(int*, int);
int __hpvm__atomic_umax(int*, int);
int __hpvm__atomic_umin(int*, int);
int __hpvm__atomic_and(int*, int);
int __hpvm__atomic_or(int*, int);
int __hpvm__atomic_xor(int*, int);

// Special Func
float __hpvm__floor(float);
float __hpvm__rsqrt(float);
float __hpvm__sqrt(float);
float __hpvm__sin(float);

int get_global_id(int);
int get_group_id(int);
int get_local_id(int);
int get_local_size(int);

void llvm_hpvm_track_mem(void*, size_t);
void llvm_hpvm_untrack_mem(void*);
void llvm_hpvm_request_mem(void*, size_t);

