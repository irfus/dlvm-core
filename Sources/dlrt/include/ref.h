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

/**
 Reference
 */
struct _dl_ref {
    void (* _Nonnull free)(const struct _dl_ref * const _Nonnull);
    _Atomic long count;
} __attribute__((swift_name("DLReference")));

struct _dl_ref _dl_ref_init(void (* _Nonnull free)(const struct _dl_ref * const _Nonnull))
    __attribute__((swift_name("DLReference.init(free:)")));

void _dl_ref_retain(struct _dl_ref * const _Nonnull ref)
    __attribute__((swift_name("DLReference.retain(self:)")));

void _dl_ref_release(struct _dl_ref * const _Nonnull ref)
    __attribute__((swift_name("DLReference.release(self:)")));

void _dl_ref_dealloc(struct _dl_ref * const _Nonnull ref)
    __attribute__((swift_name("DLReference.deallocate(self:)")));

long _dl_ref_count(const struct _dl_ref * const _Nonnull ref)
    __attribute__((swift_name("getter:DLReference.count(self:)")));
