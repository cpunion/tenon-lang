// std/interop/objc_shim.m — Dynamic Apple Objective-C / Swift `@objc` Object Interop (macOS / iOS!)
// Enables Tenon to seamlessly instantiate, message, and inspect native Foundation / CoreML / Swift `@objc` objects.

#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <objc/runtime.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern int64_t tenon_text_new(int64_t data_ptr, int64_t byte_len);

int64_t tn_objc_class(int64_t name_cstr) {
    if (!name_cstr) return 0;
    Class cls = objc_getClass((const char*)(uintptr_t)name_cstr);
    return (int64_t)(uintptr_t)cls;
}

int64_t tn_objc_nsstring(int64_t cstr, int64_t len) {
    @autoreleasepool {
        NSString* s = [[NSString alloc] initWithBytes:(const void*)(uintptr_t)cstr
                                               length:(NSUInteger)len
                                             encoding:NSUTF8StringEncoding];
        return (int64_t)(uintptr_t)CFBridgingRetain(s);
    }
}

int64_t tn_objc_to_text(int64_t obj_h) {
    if (!obj_h) return tenon_text_new(0, 0);
    @autoreleasepool {
        id obj = (__bridge id)(void*)(uintptr_t)obj_h;
        NSString* desc = [obj description];
        const char* utf8 = [desc UTF8String];
        return utf8 ? tenon_text_new((int64_t)(uintptr_t)utf8, (int64_t)strlen(utf8)) : tenon_text_new(0, 0);
    }
}

void tn_objc_release(int64_t obj_h) {
    if (obj_h) CFRelease((CFTypeRef)(uintptr_t)obj_h);
}

int64_t tn_objc_send0_obj(int64_t target_h, int64_t sel_cstr) {
    if (!target_h || !sel_cstr) return 0;
    @autoreleasepool {
        id target = (__bridge id)(void*)(uintptr_t)target_h;
        SEL sel = sel_registerName((const char*)(uintptr_t)sel_cstr);
        id res = ((id (*)(id, SEL))objc_msgSend)(target, sel);
        return res ? (int64_t)(uintptr_t)CFBridgingRetain(res) : 0;
    }
}

int64_t tn_objc_send0_int(int64_t target_h, int64_t sel_cstr) {
    if (!target_h || !sel_cstr) return 0;
    @autoreleasepool {
        id target = (__bridge id)(void*)(uintptr_t)target_h;
        SEL sel = sel_registerName((const char*)(uintptr_t)sel_cstr);
        return (int64_t)((NSInteger (*)(id, SEL))objc_msgSend)(target, sel);
    }
}

int64_t tn_objc_send1_obj(int64_t target_h, int64_t sel_cstr, int64_t arg_h) {
    if (!target_h || !sel_cstr) return 0;
    @autoreleasepool {
        id target = (__bridge id)(void*)(uintptr_t)target_h;
        id arg = arg_h ? (__bridge id)(void*)(uintptr_t)arg_h : nil;
        SEL sel = sel_registerName((const char*)(uintptr_t)sel_cstr);
        id res = ((id (*)(id, SEL, id))objc_msgSend)(target, sel, arg);
        return res ? (int64_t)(uintptr_t)CFBridgingRetain(res) : 0;
    }
}

int64_t tn_objc_send1_int(int64_t target_h, int64_t sel_cstr, int64_t arg) {
    if (!target_h || !sel_cstr) return 0;
    @autoreleasepool {
        id target = (__bridge id)(void*)(uintptr_t)target_h;
        SEL sel = sel_registerName((const char*)(uintptr_t)sel_cstr);
        return ((int64_t (*)(id, SEL, int64_t))objc_msgSend)(target, sel, arg);
    }
}

/* Dynamically subclass ANY Objective-C / Swift `@objc` class at runtime and bind a native
 * LLVM-compiled `@export` Tenon function `(self_h: Int, arg: Int) -> Int` as a real ObjC method! */
typedef int64_t (*tn_objc_export_fn2)(int64_t, int64_t);

int64_t tn_objc_subclass_with_method(int64_t base_name_cstr, int64_t new_name_cstr, int64_t sel_cstr, int64_t c_sym_cstr) {
    if (!base_name_cstr || !new_name_cstr || !sel_cstr || !c_sym_cstr) return 0;
    Class base = objc_getClass((const char*)(uintptr_t)base_name_cstr);
    if (!base) return 0;
    const char* new_name = (const char*)(uintptr_t)new_name_cstr;
    void* raw_sym = dlsym(RTLD_DEFAULT, (const char*)(uintptr_t)c_sym_cstr);
    if (!raw_sym) {
        fprintf(stderr, "[tenon-objc] dlsym could not resolve @export symbol `%s`\n", (const char*)(uintptr_t)c_sym_cstr);
        return 0;
    }
    tn_objc_export_fn2 target_fn = (tn_objc_export_fn2)raw_sym;
    Class sub = objc_getClass(new_name);
    if (!sub) {
        sub = objc_allocateClassPair(base, new_name, 0);
        if (!sub) return 0;
        SEL sel = sel_registerName((const char*)(uintptr_t)sel_cstr);
        IMP imp = imp_implementationWithBlock(^int64_t(id self_obj, int64_t arg) {
            return target_fn((int64_t)(uintptr_t)self_obj, arg);
        });
        class_addMethod(sub, sel, imp, "q@:q");
        objc_registerClassPair(sub);
    }
    return (int64_t)(uintptr_t)sub;
}
