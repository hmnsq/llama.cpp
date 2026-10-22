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

সংরচনা dl_handle_deleter {
    শূন্য operator()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

স্থির অন্তঃস্থ dl_handle * dl_load_library(ধ্রুব fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    প্রত্যাগম handle;
}

স্থির অন্তঃস্থ শূন্য * dl_get_sym(dl_handle * handle, ধ্রুব বর্ণ * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    শূন্য * p = (শূন্য *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    প্রত্যাগম p;
}

স্থির অন্তঃস্থ ধ্রুব বর্ণ * dl_error() {
    প্রত্যাগম "";
}

#অন্যথা

using dl_handle = শূন্য;

সংরচনা dl_handle_deleter {
    শূন্য operator()(শূন্য * handle) {
        dlclose(handle);
    }
};

স্থির অন্তঃস্থ dl_handle * dl_load_library(ধ্রুব fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    প্রত্যাগম handle;
}

স্থির অন্তঃস্থ শূন্য * dl_get_sym(dl_handle * handle, ধ্রুব বর্ণ * name) {
    প্রত্যাগম dlsym(handle, name);
}

স্থির অন্তঃস্থ ধ্রুব বর্ণ * dl_error() {
    ধ্রুব বর্ণ *rslt = dlerror();
    প্রত্যাগম rslt != nullptr ? rslt : "";
}

#endif
