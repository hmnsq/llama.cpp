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

#وگرنه

using dl_handle = تهی;

ساختار dl_handle_deleter {
    تهی operator()(تهی * handle) {
        dlclose(handle);
    }
};

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

dl_handle * dl_load_library(ثابت fs::path & path);
تهی * dl_get_sym(dl_handle * handle, ثابت نویسه * name);
ثابت نویسه * dl_error();

