//
//  swiftdecl.h
//  DLVM Comptue Primitives
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

#ifndef _SWIFTDECL_H_
#define _SWIFTDECL_H_

/// Swift name attribute with quoted expansion
#ifndef SWIFT_NAME
    #ifdef __swift__
        #define SWIFT_NAME(x) __attribute__((swift_name(#x)))
    #else
        #define SWIFT_NAME(x)
    #endif
#endif

/// Swift name attribute with direct expansion
#ifndef SWIFT_COMPILE_NAME
    #ifdef __swift__
        #define SWIFT_COMPILE_NAME(X) __attribute__((swift_name(X)))
    #else
        #define SWIFT_COMPILE_NAME(X)
    #endif
#endif

/// Swift enum attributes are available only in Objective-C mode
#ifdef __OBJC__

#ifndef SWIFT_ENUM_EXTRA
    #define SWIFT_ENUM_EXTRA
#endif

#ifndef SWIFT_ENUM
    #define SWIFT_ENUM(_type, _name) enum _name : _type _name; enum SWIFT_ENUM_EXTRA _name : _type
#endif

#ifndef SWIFT_ENUM_NAMED
    #define SWIFT_ENUM_NAMED(_type, _name, SWIFT_NAME) enum _name : _type _name SWIFT_COMPILE_NAME(SWIFT_NAME); enum SWIFT_COMPILE_NAME(SWIFT_NAME) SWIFT_ENUM_EXTRA _name : _type
#endif

#endif // __OBJC__

#endif /* _SWIFTDECL_H_ */
