//
//  access_owner.h
//  DLVM
//
//  Copyright 2016-2017 DLVM Team.
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

#ifndef _ACCESS_OWNER_H_
#define _ACCESS_OWNER_H_

#include "swiftdecl.h"
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef SWIFT_ENUM(int, DLAccessOwner) {
    host = 1,
    device = 2
};

// Runtime routines for device memory management
typedef struct DLDeviceRuntimeRoutines {
    int64_t (* const _Nonnull allocate)(void *_Nonnull *_Nullable ptr, size_t size);
    int64_t (* const _Nonnull deallocate)(void *_Nonnull);
    int64_t (* const _Nonnull copyToDevice)(void *_Nonnull, const void *_Nonnull);
    int64_t (* const _Nonnull copyToHost)(void *_Nonnull, const void *_Nonnull);
} DLDeviceRuntimeRoutines;

#ifdef __cplusplus
}
#endif

#endif /* _ACCESS_OWNER_H_ */
