#pragma once

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#അല്ലെങ്കിൽ
#    include <dlfcn.h>
#    include <unistd.h>
#endif
#അന്തര്ഭാവ <filesystem>

namespace fs = std::filesystem;

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

ഘടന dl_handle_deleter {
    ശൂന്യം operator()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

സ്ഥിതം വരിയിൽ dl_handle * dl_load_library(സ്ഥിരം fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    തിരികെ handle;
}

സ്ഥിതം വരിയിൽ ശൂന്യം * dl_get_sym(dl_handle * handle, സ്ഥിരം അക്ഷരം * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    ശൂന്യം * p = (ശൂന്യം *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    തിരികെ p;
}

സ്ഥിതം വരിയിൽ സ്ഥിരം അക്ഷരം * dl_error() {
    തിരികെ "";
}

#അല്ലെങ്കിൽ

using dl_handle = ശൂന്യം;

ഘടന dl_handle_deleter {
    ശൂന്യം operator()(ശൂന്യം * handle) {
        dlclose(handle);
    }
};

സ്ഥിതം വരിയിൽ dl_handle * dl_load_library(സ്ഥിരം fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    തിരികെ handle;
}

സ്ഥിതം വരിയിൽ ശൂന്യം * dl_get_sym(dl_handle * handle, സ്ഥിരം അക്ഷരം * name) {
    തിരികെ dlsym(handle, name);
}

സ്ഥിതം വരിയിൽ സ്ഥിരം അക്ഷരം * dl_error() {
    സ്ഥിരം അക്ഷരം *rslt = dlerror();
    തിരികെ rslt != nullptr ? rslt : "";
}

#endif
