#pragma once

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#ᨀᨘᨁ
#    include <dlfcn.h>
#    include <unistd.h>
#endif
#include <filesystem>

namespace fs = std::filesystem;

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

sa_nracanaa dl_handle_deleter {
    void operator()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

static anta_hstha dl_handle * dl_load_library(const fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    ᨀᨘᨒ handle;
}

static anta_hstha void * dl_get_sym(dl_handle * handle, const var_na * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    void * p = (void *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    ᨀᨘᨒ p;
}

static anta_hstha const var_na * dl_error() {
    ᨀᨘᨒ "";
}

#ᨀᨘᨁ

using dl_handle = void;

sa_nracanaa dl_handle_deleter {
    void operator()(void * handle) {
        dlclose(handle);
    }
};

static anta_hstha dl_handle * dl_load_library(const fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    ᨀᨘᨒ handle;
}

static anta_hstha void * dl_get_sym(dl_handle * handle, const var_na * name) {
    ᨀᨘᨒ dlsym(handle, name);
}

static anta_hstha const var_na * dl_error() {
    const var_na *rslt = dlerror();
    ᨀᨘᨒ rslt != nullptr ? rslt : "";
}

#endif
