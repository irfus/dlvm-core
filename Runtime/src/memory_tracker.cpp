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

void DLMemoryTracker::requestMemory(void * _Nonnull ptr, DLAccessOwner requester) {
    auto propsSearch = _registry.find(ptr);
    assert(propsSearch != _registry.end() && "Address is not registered");
    auto &props = propsSearch->second;
    // If datum is synchronized, do nothing
    if (!props.outOfSync)
        return;
    // H2D
    if (props.getAccessOwner() == ::host && requester == ::device) {
        auto mallocStatus = _runtimeRoutines.allocate(&props.deviceAddress, props.size);
        assert(mallocStatus == 0 && "Device allocation failed");
        auto copyStatus = _runtimeRoutines.copyToDevice(props.deviceAddress, ptr);
        assert(copyStatus == 0 && "H2D copy failed");
        props.outOfSync = false;
    }
    // D2H
    else if (props.getAccessOwner() == ::device && requester == ::host) {
        auto copyStatus = _runtimeRoutines.copyToHost(ptr, props.deviceAddress);
        assert(copyStatus == 0 && "D2H copy failed");
        auto deallocStatus = _runtimeRoutines.deallocate(props.deviceAddress);
        props.deviceAddress = nullptr;
        assert(deallocStatus == 0 && "Device deallocation failed");
        props.outOfSync = false;
    }
}

void DLMemoryTracker::registerMemory(const void * _Nonnull ptr, size_t size) {
    _registry[ptr] = { size, nullptr, false };
}

void DLMemoryTracker::unregisterMemory(const void * _Nonnull ptr) {
    _registry.erase(ptr);
}

void DLMemoryTracker::setOutOfSync(const void * _Nonnull ptr) {
    _registry[ptr].outOfSync = true;
}

void DLMemoryTracker::switchToHost(const void * _Nonnull ptr) {
    auto &props = _registry[ptr];
    if (props.deviceAddress != nullptr) {
        _runtimeRoutines.deallocate(props.deviceAddress);
        props.deviceAddress = nullptr;
    }
    props.outOfSync = false;
}

void DLMemoryTracker::clear() {
    for (auto &[ptr, props] : _registry) {
        if (props.deviceAddress != nullptr) {
            _runtimeRoutines.deallocate(props.deviceAddress);
        }
    }
    _registry.clear();
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

void DLMemoryTrackerSetOutOfSync(DLMemoryTracker * _Nonnull tracker,
                                 const void * _Nonnull ptr) {
    tracker->setOutOfSync(ptr);
}

void DLMemoryTrackerClear(DLMemoryTracker * _Nonnull tracker) {
    tracker->clear();
}

}
