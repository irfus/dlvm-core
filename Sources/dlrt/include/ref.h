//
//  ref.h
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

#import <stdatomic.h>

#define SWIFT_NAME(X) __attribute__((swift_name(#X)))

typedef enum _dl_access_owner {
    _DL_ACCESS_OWNER_HOST = 0,
    _DL_ACCESS_OWNER_DEVICE = 1
} _dl_access_owner;

/**
 Reference
 */
typedef struct _dl_ref {
    void (* _Nonnull free)(const struct _dl_ref * const _Nonnull);
    _Atomic long count;
    _dl_access_owner access_owner;
} _dl_ref __attribute__((swift_name("DLReference")));

_dl_ref _dl_ref_init(void (* _Nonnull free)(const _dl_ref * const _Nonnull))
    SWIFT_NAME(DLReference.init(free:));

void _dl_ref_retain(_dl_ref * const _Nonnull ref)
    SWIFT_NAME(DLReference.retain(self:));

void _dl_ref_release(_dl_ref * const _Nonnull ref)
    SWIFT_NAME(DLReference.release(self:));

void _dl_ref_dealloc(_dl_ref * const _Nonnull ref)
    SWIFT_NAME(DLReference.deallocate(self:));

long _dl_ref_count(const _dl_ref * const _Nonnull ref)
    SWIFT_NAME(getter:DLReference.count(self:));

_dl_access_owner _dl_ref_access_owner(_dl_ref * const _Nonnull ref)
    SWIFT_NAME(getter:DLReference.owner(self:));

void _dl_ref_transfer_access(_dl_ref * const _Nonnull ref, _dl_access_owner target)
    SWIFT_NAME(setter:DLReference.owner(self:_:));
