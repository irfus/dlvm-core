//
//  reference.c
//  DLVM
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

#include "reference.h"

DLReference DLReferenceInit(const void (* _Nonnull deallocator)(const DLReference *const _Nonnull)) {
    return (DLReference) { deallocator, 1 };
}

void DLReferenceRetain(DLReference *const _Nonnull ref)
{
    atomic_fetch_add(&ref->count, 1);
}

void DLReferenceRelease(DLReference *const _Nonnull ref)
{
    if (atomic_fetch_sub(&ref->count, 1) == 1)
        ref->deallocator(ref);
}

void DLReferenceDeallocate(DLReference *const _Nonnull ref)
{
    atomic_store(&ref->count, 0);
    ref->deallocator(ref);
}

long DLReferenceCount(const DLReference *const _Nonnull ref)
{
    return ref->count;
}
