//
//  ad_stack.h
//  DLVM
//
//  Copyright 2016-2018 The DLVM Team.
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

#ifndef _AD_STACK_H_
#define _AD_STACK_H_

#include <stack>

extern "C" {

struct DLValue {
    int64_t data;
    int8_t * _Nonnull typeMetadata;
};

typedef std::stack<DLValue> DLADStack;

DLADStack * _Nonnull DLADStackCreate();
void DLADStackDestroy(DLADStack * _Nonnull stack);
void DLADStackPush(DLADStack * _Nonnull stack, DLValue value);
void DLADStackPop(DLADStack * _Nonnull stack);
DLValue DLADStackTop(DLADStack * _Nonnull stack);
size_t DLADStackSize(DLADStack * _Nonnull stack);

}

#endif /* _AD_STACK_H_ */
