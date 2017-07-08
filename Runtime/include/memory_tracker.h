//
//  memory_tracker.h
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

#import "access_owner.h"
#import <unordered_map>

extern "C" {

struct DLDatumProperties {
    size_t size;
    void * _Nonnull deviceAddress = nullptr;
    bool outOfSync = false;

    DLAccessOwner getAccessOwner() {
        return deviceAddress == nullptr ? ::host : ::device;
    }
};

class DLMemoryTracker {
private:
    typedef std::unordered_map<const void *, DLDatumProperties> Registry;
    Registry _registry = {};
    DLDeviceRuntimeRoutines _runtimeRoutines;

public:
    ~DLMemoryTracker();
    DLMemoryTracker(DLDeviceRuntimeRoutines runtimeRoutines);
    DLMemoryTracker(const DLMemoryTracker &) = delete;
    DLMemoryTracker operator=(const DLMemoryTracker &) = delete;

    const Registry &getRegistry() {
        return _registry;
    }

    DLDatumProperties getProperties(void * _Nonnull ptr);

    void requestMemory(void * _Nonnull ptr, DLAccessOwner requester);
    void registerMemory(const void * _Nonnull ptr, size_t size);
    void unregisterMemory(const void * _Nonnull ptr);
    void setOutOfSync(const void * _Nonnull ptr);
    void switchToHost(const void * _Nonnull ptr);
    void clear();
};

DLMemoryTracker * _Nonnull DLMemoryTrackerCreate(DLDeviceRuntimeRoutines rtr);
void DLMemoryTrackerDestroy(DLMemoryTracker * _Nonnull tracker);
void DLMemoryTrackerRequestMemory(DLMemoryTracker * _Nonnull tracker,
                                  void * _Nonnull ptr, DLAccessOwner requester);
void DLMemoryTrackerRegisterMemory(DLMemoryTracker * _Nonnull tracker,
                                   const void * _Nonnull ptr, size_t size);
void DLMemoryTrackerUnregisterMemory(DLMemoryTracker * _Nonnull tracker,
                                     const void * _Nonnull ptr);

}
