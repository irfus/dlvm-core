//
//  ref.c
//  DLVM
//
//  Created by Richard Wei on 3/28/17.
//
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
