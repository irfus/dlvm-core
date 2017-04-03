//
//  ref.h
//  DLVM
//
//  Created by Richard Wei on 3/28/17.
//
//

#import <stdatomic.h>

/**
 Reference
 */
struct _dl_ref {
    void (* _Nonnull free)(const struct _dl_ref * const _Nonnull);
    _Atomic long count;
} __attribute__((swift_name("DLReference")));

struct _dl_ref _dl_ref_init(void (* __nonnull free)(const struct _dl_ref * const _Nonnull))
    __attribute__((swift_name("DLReference.init(free:)")));

void _dl_ref_retain(struct _dl_ref * const _Nonnull ref)
    __attribute__((swift_name("DLReference.retain(self:)")));

void _dl_ref_release(struct _dl_ref * const _Nonnull ref)
    __attribute__((swift_name("DLReference.release(self:)")));

void _dl_ref_dealloc(struct _dl_ref * const _Nonnull ref)
    __attribute__((swift_name("DLReference.deallocate(self:)")));

long _dl_ref_count(const struct _dl_ref * const _Nonnull ref)
    __attribute__((swift_name("getter:DLReference.count(self:)")));
