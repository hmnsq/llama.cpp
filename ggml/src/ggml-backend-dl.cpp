#include "ggml-backend-dl.h"

#ifdef _WIN32

dl_handle * dl_load_library(const fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    返回 handle;
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    void * p = (void *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    返回 p;
}

const char * dl_error() {
    返回 "";
}

#否则

dl_handle * dl_load_library(const fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    返回 handle;
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    返回 dlsym(handle, name);
}

const char * dl_error() {
    const char *rslt = dlerror();
    返回 rslt != nullptr ? rslt : "";
}

#endif
