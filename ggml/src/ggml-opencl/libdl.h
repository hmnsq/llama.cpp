#pragma once

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#ܐܚܪܢܐ
#    include <dlfcn.h>
#    include <unistd.h>
#endif
#ضمن <filesystem>

namespace fs = std::filesystem;

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

بنية dl_handle_deleter {
    فراغ operator()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

ساكن مضمن dl_handle * dl_load_library(ثابت fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    ܦܢܐ handle;
}

ساكن مضمن فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    فراغ * p = (فراغ *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    ܦܢܐ p;
}

ساكن مضمن ثابت حرف * dl_error() {
    ܦܢܐ "";
}

#ܐܚܪܢܐ

using dl_handle = فراغ;

بنية dl_handle_deleter {
    فراغ operator()(فراغ * handle) {
        dlclose(handle);
    }
};

ساكن مضمن dl_handle * dl_load_library(ثابت fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    ܦܢܐ handle;
}

ساكن مضمن فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    ܦܢܐ dlsym(handle, name);
}

ساكن مضمن ثابت حرف * dl_error() {
    ثابت حرف *rslt = dlerror();
    ܦܢܐ rslt != nullptr ? rslt : "";
}

#endif
