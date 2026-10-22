#pragma once

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#وگرنه
#    include <dlfcn.h>
#    include <unistd.h>
#endif
#شامل <filesystem>

namespace fs = std::filesystem;

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

ساختار dl_handle_deleter {
    تهی operator()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

ایستا درونخطی dl_handle * dl_load_library(ثابت fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    بازگشت handle;
}

ایستا درونخطی تهی * dl_get_sym(dl_handle * handle, ثابت نویسه * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    تهی * p = (تهی *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    بازگشت p;
}

ایستا درونخطی ثابت نویسه * dl_error() {
    بازگشت "";
}

#وگرنه

using dl_handle = تهی;

ساختار dl_handle_deleter {
    تهی operator()(تهی * handle) {
        dlclose(handle);
    }
};

ایستا درونخطی dl_handle * dl_load_library(ثابت fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    بازگشت handle;
}

ایستا درونخطی تهی * dl_get_sym(dl_handle * handle, ثابت نویسه * name) {
    بازگشت dlsym(handle, name);
}

ایستا درونخطی ثابت نویسه * dl_error() {
    ثابت نویسه *rslt = dlerror();
    بازگشت rslt != nullptr ? rslt : "";
}

#endif
