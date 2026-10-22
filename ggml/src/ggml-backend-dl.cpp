#include "ggml-backend-dl.h"

#ifdef _WIN32

dl_handle * dl_load_library(const fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    ᬯᬮᬶ handle;
}

void * dl_get_sym(dl_handle * handle, const var_na * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    void * p = (void *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    ᬯᬮᬶ p;
}

const var_na * dl_error() {
    ᬯᬮᬶ "";
}

#ᬮᬶᬬ

dl_handle * dl_load_library(const fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    ᬯᬮᬶ handle;
}

void * dl_get_sym(dl_handle * handle, const var_na * name) {
    ᬯᬮᬶ dlsym(handle, name);
}

const var_na * dl_error() {
    const var_na *rslt = dlerror();
    ᬯᬮᬶ rslt != nullptr ? rslt : "";
}

#endif
