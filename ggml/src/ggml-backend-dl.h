#pragma once

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#ಅನ್ಯಥಾ
#    include <dlfcn.h>
#    include <unistd.h>
#endif
#ಅನ್ತರ್ಭಾವ <filesystem>

namespace fs = std::filesystem;

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

ಸಂರಚನಾ dl_handle_deleter {
    ಶೂನ್ಯ operator()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

#ಅನ್ಯಥಾ

using dl_handle = ಶೂನ್ಯ;

ಸಂರಚನಾ dl_handle_deleter {
    ಶೂನ್ಯ operator()(ಶೂನ್ಯ * handle) {
        dlclose(handle);
    }
};

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

dl_handle * dl_load_library(ಧ್ರುವ fs::path & path);
ಶೂನ್ಯ * dl_get_sym(dl_handle * handle, ಧ್ರುವ ವರ್ಣ * name);
ಧ್ರುವ ವರ್ಣ * dl_error();

