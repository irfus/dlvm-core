//
//  ref.c
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

#import "ref.h"

struct _dl_ref _dl_ref_init(void (* _Nonnull free)(const struct _dl_ref * const _Nonnull)) {
    return (struct _dl_ref){ free, 1 };
}

void _dl_ref_retain(struct _dl_ref *const _Nonnull ref)
{
    atomic_fetch_add((_Atomic int *)&ref->count, 1);
}

void _dl_ref_release(struct _dl_ref *const _Nonnull ref)
{
    if (atomic_fetch_sub((_Atomic int *)&ref->count, 1) == 1)
        ref->free(ref);
}

void _dl_ref_dealloc(struct _dl_ref *const _Nonnull ref)
{
    ref->free(ref);
}

long _dl_ref_count(const struct _dl_ref *const _Nonnull ref)
{
    return ref->count;
}
