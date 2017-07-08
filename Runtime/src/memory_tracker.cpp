//
//  memory_tracker.cpp
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

#import <cassert>
#import "access_owner.h"
#import "memory_tracker.h"

DLMemoryTracker::DLMemoryTracker(DLDeviceRuntimeRoutines runtimeRoutines)
    : _runtimeRoutines(runtimeRoutines) {
}

DLMemoryTracker::~DLMemoryTracker() {
}

DLDatumProperties DLMemoryTracker::getProperties(void * _Nonnull ptr) {
    auto search = _registry.find(ptr);
    assert(search != _registry.end() && "Address is not registered");
    return search->second;
}

void *DLMemoryTracker::getDeviceAddress(void * _Nonnull ptr) {
    auto search = _memoryMap.find(ptr);
    assert(search != _memoryMap.end() && "Data is not on device");
    return search->second;
}

void DLMemoryTracker::requestMemory(void * _Nonnull ptr, DLAccessOwner requester) {
    auto ownerSearch = _registry.find(ptr);
    assert(ownerSearch != _registry.end() && "Address is not registered");
    auto props = ownerSearch->second;
    // H2D
    if (props.owner == ::host && requester == ::device) {
        void *devAddr = nullptr;
        auto mallocStatus = _runtimeRoutines.allocate(&devAddr, props.size);
        assert(mallocStatus == 0 && "Device allocation failed");
        auto copyStatus = _runtimeRoutines.copyToDevice(devAddr, ptr);
        _memoryMap[ptr] = devAddr;
        assert(copyStatus == 0 && "H2D copy failed");
    }
    // D2H
    else if (props.owner == ::device && requester == ::host) {
        auto devAddr = getDeviceAddress(ptr);
        auto copyStatus = _runtimeRoutines.copyToHost(ptr, devAddr);
        assert(copyStatus == 0 && "D2H copy failed");
        _memoryMap.erase(ptr);
        auto deallocStatus = _runtimeRoutines.deallocate(devAddr);
        assert(deallocStatus == 0 && "Device deallocation failed");
    }
}

void DLMemoryTracker::registerMemory(const void * _Nonnull ptr, size_t size) {
    _registry[ptr] = { size, ::host };
}

void DLMemoryTracker::unregisterMemory(const void * _Nonnull ptr) {
    _registry.erase(ptr);
}

void DLMemoryTracker::clear() {
    _registry.clear();
    _memoryMap.clear();
}

#pragma mark - Exported API

extern "C" {

DLMemoryTracker * _Nonnull DLMemoryTrackerCreate(DLDeviceRuntimeRoutines rtr) {
    return new DLMemoryTracker(rtr);
}

void DLMemoryTrackerDestroy(DLMemoryTracker * _Nonnull tracker) {
    delete tracker;
}

void DLMemoryTrackerRequestMemory(DLMemoryTracker * _Nonnull tracker,
                                  void * _Nonnull ptr, DLAccessOwner requester) {
    tracker->requestMemory(ptr, requester);
}

void DLMemoryTrackerRegisterMemory(DLMemoryTracker * _Nonnull tracker,
                                   const void * _Nonnull ptr, size_t size) {
    tracker->registerMemory(ptr, size);
}

void DLMemoryTrackerUnregisterMemory(DLMemoryTracker * _Nonnull tracker,
                                     const void * _Nonnull ptr) {
    tracker->unregisterMemory(ptr);
}

void DLMemoryTrackerClear(DLMemoryTracker * _Nonnull tracker) {
    tracker->clear();
}

}
