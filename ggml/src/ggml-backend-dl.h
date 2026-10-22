#pragma once

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#དེ་མིན
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

#དེ་མིན

using dl_handle = void;

sa_nracanaa dl_handle_deleter {
    void operator()(void * handle) {
        dlclose(handle);
    }
};

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

dl_handle * dl_load_library(const fs::path & path);
void * dl_get_sym(dl_handle * handle, const var_na * name);
const var_na * dl_error();

