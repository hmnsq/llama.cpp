#pragma once

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#অন্যথা
#    include <dlfcn.h>
#    include <unistd.h>
#endif
#অন্তর্ভাব <filesystem>

namespace fs = std::filesystem;

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

কাঠামো dl_handle_deleter {
    শূন্য operator()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

স্থির অন্তঃরেখ dl_handle * dl_load_library(ধ্রুবক fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    ফেরত handle;
}

স্থির অন্তঃরেখ শূন্য * dl_get_sym(dl_handle * handle, ধ্রুবক অক্ষর * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    শূন্য * p = (শূন্য *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    ফেরত p;
}

স্থির অন্তঃরেখ ধ্রুবক অক্ষর * dl_error() {
    ফেরত "";
}

#অন্যথা

using dl_handle = শূন্য;

কাঠামো dl_handle_deleter {
    শূন্য operator()(শূন্য * handle) {
        dlclose(handle);
    }
};

স্থির অন্তঃরেখ dl_handle * dl_load_library(ধ্রুবক fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    ফেরত handle;
}

স্থির অন্তঃরেখ শূন্য * dl_get_sym(dl_handle * handle, ধ্রুবক অক্ষর * name) {
    ফেরত dlsym(handle, name);
}

স্থির অন্তঃরেখ ধ্রুবক অক্ষর * dl_error() {
    ধ্রুবক অক্ষর *rslt = dlerror();
    ফেরত rslt != nullptr ? rslt : "";
}

#endif
