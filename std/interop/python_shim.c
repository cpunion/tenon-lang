// std/interop/python_shim.c — Mojo-Grade Dynamic CPython 3 Bridge + Native Tenon Subclass Extension!
//
// Zero build-flag friction: dynamically `dlopen`s `libpython3` / `Python.framework` at runtime with
// `RTLD_GLOBAL` so CPython C extensions (math, json, numpy, torch, pandas) load cleanly!
// Supports:
//   1. Seamless Python module import, dynamic attribute/method dispatch, eval/exec
//   2. Zero-copy `PyMemoryView_FromMemory` sharing of raw Tenon buffers/tensors with Python/NumPy/PyTorch
//   3. Extending & subclassing ANY Python class with native LLVM-compiled Tenon functions (`PyCFunction`!)

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int64_t tenon_text_new(int64_t data_ptr, int64_t byte_len);

/* Opaque CPython ABI declarations (stable across Python 3.9 - 3.14+!) */
typedef void PyObject;
typedef intptr_t Py_ssize_t;
typedef PyObject* (*PyCFunction)(PyObject* self, PyObject* args);

typedef struct {
    const char* ml_name;
    PyCFunction ml_meth;
    int ml_flags;
    const char* ml_doc;
} PyMethodDef;

#define METH_VARARGS 0x0001
#define PyBUF_READ   0x100
#define PyBUF_WRITE  0x200

typedef void (*fn_Py_InitializeEx)(int);
typedef int (*fn_Py_IsInitialized)(void);
typedef void (*fn_Py_IncRef)(PyObject*);
typedef void (*fn_Py_DecRef)(PyObject*);
typedef PyObject* (*fn_PyImport_ImportModule)(const char*);
typedef PyObject* (*fn_PyImport_AddModule)(const char*);
typedef PyObject* (*fn_PyModule_GetDict)(PyObject*);
typedef PyObject* (*fn_PyEval_GetBuiltins)(void);
typedef int (*fn_PyRun_SimpleString)(const char*);
typedef PyObject* (*fn_PyRun_String)(const char*, int, PyObject*, PyObject*);
typedef PyObject* (*fn_PyObject_GetAttrString)(PyObject*, const char*);
typedef int (*fn_PyObject_SetAttrString)(PyObject*, const char*, PyObject*);
typedef PyObject* (*fn_PyObject_GetItem)(PyObject*, PyObject*);
typedef int (*fn_PyObject_SetItem)(PyObject*, PyObject*, PyObject*);
typedef PyObject* (*fn_PyObject_CallObject)(PyObject*, PyObject*);
typedef PyObject* (*fn_PyObject_Str)(PyObject*);
typedef Py_ssize_t (*fn_PyObject_Length)(PyObject*);
typedef PyObject* (*fn_PyLong_FromLongLong)(long long);
typedef long long (*fn_PyLong_AsLongLong)(PyObject*);
typedef PyObject* (*fn_PyFloat_FromDouble)(double);
typedef double (*fn_PyFloat_AsDouble)(PyObject*);
typedef PyObject* (*fn_PyBool_FromLong)(long);
typedef PyObject* (*fn_PyUnicode_FromStringAndSize)(const char*, Py_ssize_t);
typedef const char* (*fn_PyUnicode_AsUTF8AndSize)(PyObject*, Py_ssize_t*);
typedef PyObject* (*fn_PyTuple_New)(Py_ssize_t);
typedef int (*fn_PyTuple_SetItem)(PyObject*, Py_ssize_t, PyObject*);
typedef PyObject* (*fn_PyTuple_GetItem)(PyObject*, Py_ssize_t);
typedef Py_ssize_t (*fn_PyTuple_Size)(PyObject*);
typedef PyObject* (*fn_PyList_New)(Py_ssize_t);
typedef int (*fn_PyList_Append)(PyObject*, PyObject*);
typedef PyObject* (*fn_PyDict_New)(void);
typedef int (*fn_PyDict_SetItemString)(PyObject*, const char*, PyObject*);
typedef PyObject* (*fn_PyDict_GetItemString)(PyObject*, const char*);
typedef PyObject* (*fn_PyMemoryView_FromMemory)(char*, Py_ssize_t, int);
typedef PyObject* (*fn_PyCFunction_NewEx)(PyMethodDef*, PyObject*, PyObject*);
typedef PyObject* (*fn_PyCapsule_New)(void*, const char*, void*);
typedef void* (*fn_PyCapsule_GetPointer)(PyObject*, const char*);
typedef PyObject* (*fn_PyErr_Occurred)(void);
typedef void (*fn_PyErr_Clear)(void);
typedef void (*fn_PyErr_Print)(void);

typedef struct {
    void* lib;
    int ready;
    fn_Py_InitializeEx Py_InitializeEx;
    fn_Py_IsInitialized Py_IsInitialized;
    fn_Py_IncRef Py_IncRef;
    fn_Py_DecRef Py_DecRef;
    fn_PyImport_ImportModule PyImport_ImportModule;
    fn_PyImport_AddModule PyImport_AddModule;
    fn_PyModule_GetDict PyModule_GetDict;
    fn_PyEval_GetBuiltins PyEval_GetBuiltins;
    fn_PyRun_SimpleString PyRun_SimpleString;
    fn_PyRun_String PyRun_String;
    fn_PyObject_GetAttrString PyObject_GetAttrString;
    fn_PyObject_SetAttrString PyObject_SetAttrString;
    fn_PyObject_GetItem PyObject_GetItem;
    fn_PyObject_SetItem PyObject_SetItem;
    fn_PyObject_CallObject PyObject_CallObject;
    fn_PyObject_Str PyObject_Str;
    fn_PyObject_Length PyObject_Length;
    fn_PyLong_FromLongLong PyLong_FromLongLong;
    fn_PyLong_AsLongLong PyLong_AsLongLong;
    fn_PyFloat_FromDouble PyFloat_FromDouble;
    fn_PyFloat_AsDouble PyFloat_AsDouble;
    fn_PyBool_FromLong PyBool_FromLong;
    fn_PyUnicode_FromStringAndSize PyUnicode_FromStringAndSize;
    fn_PyUnicode_AsUTF8AndSize PyUnicode_AsUTF8AndSize;
    fn_PyTuple_New PyTuple_New;
    fn_PyTuple_SetItem PyTuple_SetItem;
    fn_PyTuple_GetItem PyTuple_GetItem;
    fn_PyTuple_Size PyTuple_Size;
    fn_PyList_New PyList_New;
    fn_PyList_Append PyList_Append;
    fn_PyDict_New PyDict_New;
    fn_PyDict_SetItemString PyDict_SetItemString;
    fn_PyDict_GetItemString PyDict_GetItemString;
    fn_PyMemoryView_FromMemory PyMemoryView_FromMemory;
    fn_PyCFunction_NewEx PyCFunction_NewEx;
    fn_PyCapsule_New PyCapsule_New;
    fn_PyCapsule_GetPointer PyCapsule_GetPointer;
    fn_PyErr_Occurred PyErr_Occurred;
    fn_PyErr_Clear PyErr_Clear;
    fn_PyErr_Print PyErr_Print;
    PyObject* py_none;
} TenonPyApi;

static TenonPyApi g_py = {0};

static void* tn_try_dlopen_python(void) {
    const char* env_lib = getenv("TENON_PYTHON_LIB");
    if (env_lib && env_lib[0]) {
        void* h = dlopen(env_lib, RTLD_LAZY | RTLD_GLOBAL);
        if (h) return h;
    }
    const char* candidates[] = {
        "/opt/homebrew/opt/python@3.14/Frameworks/Python.framework/Versions/3.14/Python",
        "/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/Python",
        "/opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/Python",
        "/opt/homebrew/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/Python",
        "/Library/Frameworks/Python.framework/Versions/Current/Python",
        "/usr/local/Frameworks/Python.framework/Versions/Current/Python",
        "libpython3.14.dylib", "libpython3.13.dylib", "libpython3.12.dylib", "libpython3.dylib",
        "libpython3.14.so", "libpython3.13.so", "libpython3.12.so", "libpython3.11.so", "libpython3.so",
        NULL
    };
    for (int i = 0; candidates[i]; i++) {
        void* h = dlopen(candidates[i], RTLD_LAZY | RTLD_GLOBAL);
        if (h) return h;
    }
    /* Last resort: ask `python3` where its shared library lives */
    FILE* fp = popen("python3 -c 'import sysconfig,os; print(os.path.join(sysconfig.get_config_var(\"LIBDIR\") or \"\", sysconfig.get_config_var(\"LDLIBRARY\") or \"\"))' 2>/dev/null", "r");
    if (fp) {
        char buf[512] = {0};
        if (fgets(buf, sizeof(buf), fp)) {
            size_t n = strlen(buf);
            while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = '\0';
            if (n > 0) {
                void* h = dlopen(buf, RTLD_LAZY | RTLD_GLOBAL);
                pclose(fp);
                if (h) return h;
            }
        }
        pclose(fp);
    }
    return NULL;
}

int64_t tn_py_init(void) {
    if (g_py.ready) return 1;
    void* lib = tn_try_dlopen_python();
    if (!lib) return 0;
    g_py.lib = lib;
    #define LOAD_SYM(name) do { \
        g_py.name = (fn_##name)dlsym(lib, #name); \
        if (!g_py.name) { fprintf(stderr, "[tenon-python] missing symbol %s\n", #name); return 0; } \
    } while (0)
    LOAD_SYM(Py_InitializeEx);
    LOAD_SYM(Py_IsInitialized);
    LOAD_SYM(Py_IncRef);
    LOAD_SYM(Py_DecRef);
    LOAD_SYM(PyImport_ImportModule);
    LOAD_SYM(PyImport_AddModule);
    LOAD_SYM(PyModule_GetDict);
    LOAD_SYM(PyEval_GetBuiltins);
    LOAD_SYM(PyRun_SimpleString);
    LOAD_SYM(PyRun_String);
    LOAD_SYM(PyObject_GetAttrString);
    LOAD_SYM(PyObject_SetAttrString);
    LOAD_SYM(PyObject_GetItem);
    LOAD_SYM(PyObject_SetItem);
    LOAD_SYM(PyObject_CallObject);
    LOAD_SYM(PyObject_Str);
    LOAD_SYM(PyObject_Length);
    LOAD_SYM(PyLong_FromLongLong);
    LOAD_SYM(PyLong_AsLongLong);
    LOAD_SYM(PyFloat_FromDouble);
    LOAD_SYM(PyFloat_AsDouble);
    LOAD_SYM(PyBool_FromLong);
    LOAD_SYM(PyUnicode_FromStringAndSize);
    LOAD_SYM(PyUnicode_AsUTF8AndSize);
    LOAD_SYM(PyTuple_New);
    LOAD_SYM(PyTuple_SetItem);
    LOAD_SYM(PyTuple_GetItem);
    LOAD_SYM(PyTuple_Size);
    LOAD_SYM(PyList_New);
    LOAD_SYM(PyList_Append);
    LOAD_SYM(PyDict_New);
    LOAD_SYM(PyDict_SetItemString);
    LOAD_SYM(PyDict_GetItemString);
    LOAD_SYM(PyMemoryView_FromMemory);
    LOAD_SYM(PyCFunction_NewEx);
    LOAD_SYM(PyCapsule_New);
    LOAD_SYM(PyCapsule_GetPointer);
    LOAD_SYM(PyErr_Occurred);
    LOAD_SYM(PyErr_Clear);
    LOAD_SYM(PyErr_Print);
    #undef LOAD_SYM

    if (!g_py.Py_IsInitialized()) {
        g_py.Py_InitializeEx(0);
    }
    PyObject* main_mod = g_py.PyImport_AddModule("__main__");
    PyObject* main_dict = main_mod ? g_py.PyModule_GetDict(main_mod) : NULL;
    g_py.py_none = g_py.PyRun_String("None", 258 /* Py_eval_input */, main_dict, main_dict);
    g_py.ready = 1;
    return 1;
}

void tn_py_decref(int64_t h) {
    if (g_py.ready && h != 0) g_py.Py_DecRef((PyObject*)(uintptr_t)h);
}
int64_t tn_py_incref(int64_t h) {
    if (g_py.ready && h != 0) g_py.Py_IncRef((PyObject*)(uintptr_t)h);
    return h;
}

int64_t tn_py_import(int64_t name_cstr) {
    if (!tn_py_init() || name_cstr == 0) return 0;
    PyObject* m = g_py.PyImport_ImportModule((const char*)(uintptr_t)name_cstr);
    if (!m && g_py.PyErr_Occurred()) { g_py.PyErr_Print(); g_py.PyErr_Clear(); }
    return (int64_t)(uintptr_t)m;
}

int64_t tn_py_exec(int64_t code_cstr) {
    if (!tn_py_init() || code_cstr == 0) return 0;
    return g_py.PyRun_SimpleString((const char*)(uintptr_t)code_cstr) == 0 ? 1 : 0;
}

int64_t tn_py_eval(int64_t expr_cstr) {
    if (!tn_py_init() || expr_cstr == 0) return 0;
    PyObject* main_mod = g_py.PyImport_AddModule("__main__");
    PyObject* main_dict = g_py.PyModule_GetDict(main_mod);
    PyObject* res = g_py.PyRun_String((const char*)(uintptr_t)expr_cstr, 258 /* Py_eval_input */, main_dict, main_dict);
    if (!res && g_py.PyErr_Occurred()) { g_py.PyErr_Print(); g_py.PyErr_Clear(); }
    return (int64_t)(uintptr_t)res;
}

int64_t tn_py_none(void) {
    if (!tn_py_init()) return 0;
    g_py.Py_IncRef(g_py.py_none);
    return (int64_t)(uintptr_t)g_py.py_none;
}
int64_t tn_py_int(int64_t v) { return tn_py_init() ? (int64_t)(uintptr_t)g_py.PyLong_FromLongLong((long long)v) : 0; }
int64_t tn_py_float(double v) { return tn_py_init() ? (int64_t)(uintptr_t)g_py.PyFloat_FromDouble(v) : 0; }
int64_t tn_py_bool(int64_t v) { return tn_py_init() ? (int64_t)(uintptr_t)g_py.PyBool_FromLong(v != 0 ? 1L : 0L) : 0; }
int64_t tn_py_str(int64_t cstr, int64_t len) {
    if (!tn_py_init()) return 0;
    return (int64_t)(uintptr_t)g_py.PyUnicode_FromStringAndSize((const char*)(uintptr_t)cstr, (Py_ssize_t)len);
}
int64_t tn_py_list_new(void) { return tn_py_init() ? (int64_t)(uintptr_t)g_py.PyList_New(0) : 0; }
int64_t tn_py_list_append(int64_t lst, int64_t item) {
    if (!g_py.ready || !lst || !item) return 0;
    return g_py.PyList_Append((PyObject*)(uintptr_t)lst, (PyObject*)(uintptr_t)item) == 0 ? 1 : 0;
}
int64_t tn_py_dict_new(void) { return tn_py_init() ? (int64_t)(uintptr_t)g_py.PyDict_New() : 0; }

int64_t tn_py_to_int(int64_t h) {
    if (!g_py.ready || !h) return 0;
    long long v = g_py.PyLong_AsLongLong((PyObject*)(uintptr_t)h);
    if (g_py.PyErr_Occurred()) g_py.PyErr_Clear();
    return (int64_t)v;
}
double tn_py_to_float(int64_t h) {
    if (!g_py.ready || !h) return 0.0;
    double v = g_py.PyFloat_AsDouble((PyObject*)(uintptr_t)h);
    if (g_py.PyErr_Occurred()) g_py.PyErr_Clear();
    return v;
}
int64_t tn_py_to_text(int64_t h) {
    if (!g_py.ready || !h) return tenon_text_new(0, 0);
    PyObject* so = g_py.PyObject_Str((PyObject*)(uintptr_t)h);
    if (!so) { if (g_py.PyErr_Occurred()) g_py.PyErr_Clear(); return tenon_text_new(0, 0); }
    Py_ssize_t sz = 0;
    const char* s = g_py.PyUnicode_AsUTF8AndSize(so, &sz);
    int64_t tn = s ? tenon_text_new((int64_t)(uintptr_t)s, (int64_t)sz) : tenon_text_new(0, 0);
    g_py.Py_DecRef(so);
    return tn;
}
int64_t tn_py_len(int64_t h) {
    if (!g_py.ready || !h) return 0;
    Py_ssize_t n = g_py.PyObject_Length((PyObject*)(uintptr_t)h);
    if (n < 0 && g_py.PyErr_Occurred()) { g_py.PyErr_Clear(); return 0; }
    return (int64_t)n;
}

int64_t tn_py_getattr(int64_t obj, int64_t attr_cstr) {
    if (!g_py.ready || !obj || !attr_cstr) return 0;
    PyObject* r = g_py.PyObject_GetAttrString((PyObject*)(uintptr_t)obj, (const char*)(uintptr_t)attr_cstr);
    if (!r && g_py.PyErr_Occurred()) { g_py.PyErr_Print(); g_py.PyErr_Clear(); }
    return (int64_t)(uintptr_t)r;
}
int64_t tn_py_setattr(int64_t obj, int64_t attr_cstr, int64_t val) {
    if (!g_py.ready || !obj || !attr_cstr || !val) return 0;
    return g_py.PyObject_SetAttrString((PyObject*)(uintptr_t)obj, (const char*)(uintptr_t)attr_cstr, (PyObject*)(uintptr_t)val) == 0 ? 1 : 0;
}
int64_t tn_py_getitem(int64_t obj, int64_t key) {
    if (!g_py.ready || !obj || !key) return 0;
    PyObject* r = g_py.PyObject_GetItem((PyObject*)(uintptr_t)obj, (PyObject*)(uintptr_t)key);
    if (!r && g_py.PyErr_Occurred()) { g_py.PyErr_Clear(); }
    return (int64_t)(uintptr_t)r;
}
int64_t tn_py_setitem(int64_t obj, int64_t key, int64_t val) {
    if (!g_py.ready || !obj || !key || !val) return 0;
    return g_py.PyObject_SetItem((PyObject*)(uintptr_t)obj, (PyObject*)(uintptr_t)key, (PyObject*)(uintptr_t)val) == 0 ? 1 : 0;
}

static int64_t tn_py_call_n(int64_t fn, int n, int64_t a0, int64_t a1, int64_t a2) {
    if (!g_py.ready || !fn) return 0;
    PyObject* tup = g_py.PyTuple_New((Py_ssize_t)n);
    int64_t arr[3] = { a0, a1, a2 };
    for (int i = 0; i < n; i++) {
        PyObject* item = (PyObject*)(uintptr_t)arr[i];
        g_py.Py_IncRef(item);
        g_py.PyTuple_SetItem(tup, (Py_ssize_t)i, item);
    }
    PyObject* res = g_py.PyObject_CallObject((PyObject*)(uintptr_t)fn, tup);
    g_py.Py_DecRef(tup);
    if (!res && g_py.PyErr_Occurred()) { g_py.PyErr_Print(); g_py.PyErr_Clear(); }
    return (int64_t)(uintptr_t)res;
}
int64_t tn_py_call0(int64_t fn) { return tn_py_call_n(fn, 0, 0, 0, 0); }
int64_t tn_py_call1(int64_t fn, int64_t a0) { return tn_py_call_n(fn, 1, a0, 0, 0); }
int64_t tn_py_call2(int64_t fn, int64_t a0, int64_t a1) { return tn_py_call_n(fn, 2, a0, a1, 0); }
int64_t tn_py_call3(int64_t fn, int64_t a0, int64_t a1, int64_t a2) { return tn_py_call_n(fn, 3, a0, a1, a2); }

/* Zero-copy memoryview sharing raw Tenon memory directly with Python / NumPy / PyTorch! */
int64_t tn_py_memoryview_from_ptr(int64_t ptr, int64_t nbytes, int64_t writable) {
    if (!tn_py_init() || ptr == 0 || nbytes <= 0) return 0;
    PyObject* mv = g_py.PyMemoryView_FromMemory((char*)(uintptr_t)ptr, (Py_ssize_t)nbytes, writable ? PyBUF_WRITE : PyBUF_READ);
    if (!mv && g_py.PyErr_Occurred()) { g_py.PyErr_Print(); g_py.PyErr_Clear(); }
    return (int64_t)(uintptr_t)mv;
}

/* ===== BEYOND MOJO: Extending & Subclassing ANY Python Class with Native LLVM-Compiled Tenon Methods! ===== */
typedef struct {
    int64_t closure_or_fn;
    int arity;
    int is_method; /* 1 = first Python arg is `self` PyObject*, passed as int64_t handle to Tenon! */
    int ret_kind;  /* 0 = int64, 1 = double */
} TenonPyCallback;

typedef int64_t (*tn_cb1)(int64_t);
typedef int64_t (*tn_cb2)(int64_t, int64_t);
typedef int64_t (*tn_cb3)(int64_t, int64_t, int64_t);
typedef double (*tn_fcb1)(int64_t);
typedef double (*tn_fcb2)(int64_t, double);
typedef double (*tn_fcb3)(int64_t, double, double);

static PyObject* tn_py_trampoline(PyObject* self_cap, PyObject* args) {
    TenonPyCallback* cb = (TenonPyCallback*)g_py.PyCapsule_GetPointer(self_cap, "tenon.cb");
    if (!cb) return NULL;
    Py_ssize_t nargs = g_py.PyTuple_Size(args);
    if (cb->ret_kind == 1) {
        int64_t self_h = nargs > 0 ? (int64_t)(uintptr_t)g_py.PyTuple_GetItem(args, 0) : 0;
        double d1 = 0.0, d2 = 0.0;
        if (cb->arity >= 2 && nargs >= 2) {
            d1 = g_py.PyFloat_AsDouble(g_py.PyTuple_GetItem(args, 1));
            if (g_py.PyErr_Occurred()) g_py.PyErr_Clear();
        }
        if (cb->arity >= 3 && nargs >= 3) {
            d2 = g_py.PyFloat_AsDouble(g_py.PyTuple_GetItem(args, 2));
            if (g_py.PyErr_Occurred()) g_py.PyErr_Clear();
        }
        double fres = 0.0;
        if (cb->arity == 1) fres = ((tn_fcb1)(uintptr_t)cb->closure_or_fn)(self_h);
        else if (cb->arity == 2) fres = ((tn_fcb2)(uintptr_t)cb->closure_or_fn)(self_h, d1);
        else if (cb->arity == 3) fres = ((tn_fcb3)(uintptr_t)cb->closure_or_fn)(self_h, d1, d2);
        return g_py.PyFloat_FromDouble(fres);
    }
    int64_t vals[3] = {0, 0, 0};
    for (int i = 0; i < cb->arity && i < 3 && i < (int)nargs; i++) {
        PyObject* item = g_py.PyTuple_GetItem(args, (Py_ssize_t)i);
        if (cb->is_method && i == 0) {
            vals[i] = (int64_t)(uintptr_t)item; /* Borrowed Python `self` object handle! */
        } else {
            vals[i] = (int64_t)g_py.PyLong_AsLongLong(item);
            if (g_py.PyErr_Occurred()) {
                g_py.PyErr_Clear();
                vals[i] = (int64_t)(uintptr_t)item;
            }
        }
    }
    int64_t res = 0;
    if (cb->arity == 1) res = ((tn_cb1)(uintptr_t)cb->closure_or_fn)(vals[0]);
    else if (cb->arity == 2) res = ((tn_cb2)(uintptr_t)cb->closure_or_fn)(vals[0], vals[1]);
    else if (cb->arity == 3) res = ((tn_cb3)(uintptr_t)cb->closure_or_fn)(vals[0], vals[1], vals[2]);
    return g_py.PyLong_FromLongLong((long long)res);
}

static PyObject* tn_py_build_callable(const char* name, int64_t fn_addr, int arity, int is_method, int ret_kind) {
    TenonPyCallback* cb = (TenonPyCallback*)calloc(1, sizeof(TenonPyCallback));
    cb->closure_or_fn = fn_addr;
    cb->arity = arity;
    cb->is_method = is_method;
    cb->ret_kind = ret_kind;
    PyMethodDef* def = (PyMethodDef*)calloc(1, sizeof(PyMethodDef));
    def->ml_name = strdup(name);
    def->ml_meth = tn_py_trampoline;
    def->ml_flags = METH_VARARGS;
    def->ml_doc = "Native LLVM-compiled Tenon function/method";
    PyObject* cap = g_py.PyCapsule_New(cb, "tenon.cb", NULL);
    PyObject* cfn = g_py.PyCFunction_NewEx(def, cap, NULL);
    g_py.Py_DecRef(cap);
    return cfn;
}

/// Creates a new Python subclass `type(class_name, (base_cls,), {})` dynamically from Tenon!
int64_t tn_py_subclass_new(int64_t base_cls, int64_t class_name_cstr) {
    if (!tn_py_init() || !base_cls || !class_name_cstr) return 0;
    PyObject* builtins = g_py.PyEval_GetBuiltins();
    PyObject* py_type = g_py.PyDict_GetItemString(builtins, "type");
    if (!py_type) return 0;
    const char* cname = (const char*)(uintptr_t)class_name_cstr;
    PyObject* name_obj = g_py.PyUnicode_FromStringAndSize(cname, (Py_ssize_t)strlen(cname));
    PyObject* bases = g_py.PyTuple_New(1);
    g_py.Py_IncRef((PyObject*)(uintptr_t)base_cls);
    g_py.PyTuple_SetItem(bases, 0, (PyObject*)(uintptr_t)base_cls);
    PyObject* ns = g_py.PyDict_New();
    PyObject* args = g_py.PyTuple_New(3);
    g_py.PyTuple_SetItem(args, 0, name_obj);
    g_py.PyTuple_SetItem(args, 1, bases);
    g_py.PyTuple_SetItem(args, 2, ns);
    PyObject* sub = g_py.PyObject_CallObject(py_type, args);
    g_py.Py_DecRef(args);
    if (!sub && g_py.PyErr_Occurred()) { g_py.PyErr_Print(); g_py.PyErr_Clear(); }
    return (int64_t)(uintptr_t)sub;
}

static int64_t tn_py_attach_impl(int64_t py_cls, const char* mname, int64_t fn_addr, int arity, int ret_kind) {
    if (!g_py.ready || !py_cls || !mname || !fn_addr) return 0;
    PyObject* cfn = tn_py_build_callable(mname, fn_addr, arity, 1, ret_kind);
    if (!cfn) return 0;
    /* Capture `cfn` via immediate closure binding so attaching multiple methods never collides! */
    PyObject* main_mod = g_py.PyImport_AddModule("__main__");
    PyObject* main_dict = g_py.PyModule_GetDict(main_mod);
    g_py.PyDict_SetItemString(main_dict, "__tenon_tmp_cfn", cfn);
    g_py.Py_DecRef(cfn);
    PyObject* bound_fn = g_py.PyRun_String("(lambda _f=__tenon_tmp_cfn: (lambda *a: _f(*a)))()", 258, main_dict, main_dict);
    if (!bound_fn) { if (g_py.PyErr_Occurred()) { g_py.PyErr_Print(); g_py.PyErr_Clear(); } return 0; }
    int ok = g_py.PyObject_SetAttrString((PyObject*)(uintptr_t)py_cls, mname, bound_fn);
    g_py.Py_DecRef(bound_fn);
    return ok == 0 ? 1 : 0;
}

/// Attaches a native LLVM-compiled Tenon function directly onto ANY Python class as a real bound method!
int64_t tn_py_attach_native_method(int64_t py_cls, int64_t method_name_cstr, int64_t fn_addr, int64_t arity) {
    return tn_py_attach_impl(py_cls, (const char*)(uintptr_t)method_name_cstr, fn_addr, (int)arity, 0);
}

/// Resolves an `@export` Tenon function by symbol name via `dlsym(RTLD_DEFAULT, ...)` and attaches it as a Python method!
int64_t tn_py_attach_exported_method(int64_t py_cls, int64_t method_name_cstr, int64_t c_sym_cstr, int64_t arity) {
    if (!c_sym_cstr) return 0;
    void* sym = dlsym(RTLD_DEFAULT, (const char*)(uintptr_t)c_sym_cstr);
    if (!sym) {
        fprintf(stderr, "[tenon-python] dlsym could not resolve @export symbol `%s`\n", (const char*)(uintptr_t)c_sym_cstr);
        return 0;
    }
    return tn_py_attach_impl(py_cls, (const char*)(uintptr_t)method_name_cstr, (int64_t)(uintptr_t)sym, (int)arity, 0);
}

int64_t tn_py_attach_exported_float_method(int64_t py_cls, int64_t method_name_cstr, int64_t c_sym_cstr, int64_t arity) {
    if (!c_sym_cstr) return 0;
    void* sym = dlsym(RTLD_DEFAULT, (const char*)(uintptr_t)c_sym_cstr);
    if (!sym) {
        fprintf(stderr, "[tenon-python] dlsym could not resolve @export symbol `%s`\n", (const char*)(uintptr_t)c_sym_cstr);
        return 0;
    }
    return tn_py_attach_impl(py_cls, (const char*)(uintptr_t)method_name_cstr, (int64_t)(uintptr_t)sym, (int)arity, 1);
}

int64_t tn_py_isinstance(int64_t obj, int64_t cls) {
    if (!g_py.ready || !obj || !cls) return 0;
    PyObject* builtins = g_py.PyEval_GetBuiltins();
    PyObject* fn = g_py.PyDict_GetItemString(builtins, "isinstance");
    if (!fn) return 0;
    int64_t r = tn_py_call2((int64_t)(uintptr_t)fn, obj, cls);
    int64_t v = r ? tn_py_to_int(r) : 0;
    if (r) tn_py_decref(r);
    return v != 0 ? 1 : 0;
}

/// Exposes any Python object into Python's `__main__` global namespace under `var_name`.
int64_t tn_py_set_global(int64_t var_name_cstr, int64_t obj) {
    if (!tn_py_init() || !var_name_cstr || !obj) return 0;
    PyObject* main_mod = g_py.PyImport_AddModule("__main__");
    PyObject* main_dict = g_py.PyModule_GetDict(main_mod);
    return g_py.PyDict_SetItemString(main_dict, (const char*)(uintptr_t)var_name_cstr, (PyObject*)(uintptr_t)obj) == 0 ? 1 : 0;
}
