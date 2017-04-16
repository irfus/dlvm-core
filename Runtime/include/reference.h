//
//  reference.h
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

#import "swiftdecl.h"
#import <stdatomic.h>

typedef SWIFT_ENUM(int, DLAccessOwner) {
    DLAccessOwnerNone = 0,
    DLAccessOwnerHost,
    DLAccessOwnerDevice
};

/**
   Reference counter for DLVM IR type `@box`
*/
typedef struct DLReference {
    const void (* _Nonnull deallocator)(const struct DLReference * const _Nonnull);
    _Atomic long count;
    DLAccessOwner accessOwner;
} DLReference;

DLReference DLReferenceInit(const void (* _Nonnull deallocator)(const DLReference *const _Nonnull), const DLAccessOwner owner)
    SWIFT_NAME(DLReference.init(deallocator:owner:));

void DLReferenceRetain(DLReference *const _Nonnull ref)
    SWIFT_NAME(DLReference.retain(self:));

void DLReferenceRelease(DLReference *const _Nonnull ref)
    SWIFT_NAME(DLReference.release(self:));

void DLReferenceDeallocate(DLReference *const _Nonnull ref)
    SWIFT_NAME(DLReference.deallocate(self:));

DLAccessOwner DLReferenceGetAccessOwner(DLReference *const _Nonnull ref)
    SWIFT_NAME(getter:DLReference.owner(self:));

void DLReferenceSetAccessOwner(DLReference *const _Nonnull ref, const DLAccessOwner target)
    SWIFT_NAME(setter:DLReference.owner(self:_:));
