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

#അല്ലെങ്കിൽ

using dl_handle = ശൂന്യം;

ഘടന dl_handle_deleter {
    ശൂന്യം operator()(ശൂന്യം * handle) {
        dlclose(handle);
    }
};

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

dl_handle * dl_load_library(സ്ഥിരം fs::path & path);
ശൂന്യം * dl_get_sym(dl_handle * handle, സ്ഥിരം അക്ഷരം * name);
സ്ഥിരം അക്ഷരം * dl_error();

