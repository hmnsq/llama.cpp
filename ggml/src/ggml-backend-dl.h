#pragma once

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#ଅନ୍ୟଥା
#    include <dlfcn.h>
#    include <unistd.h>
#endif
#ଅନ୍ତର୍ଭାଵ <filesystem>

namespace fs = std::filesystem;

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

ସଂରଚନା dl_handle_deleter {
    ଶୂନ୍ୟ operator()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

#ଅନ୍ୟଥା

using dl_handle = ଶୂନ୍ୟ;

ସଂରଚନା dl_handle_deleter {
    ଶୂନ୍ୟ operator()(ଶୂନ୍ୟ * handle) {
        dlclose(handle);
    }
};

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

dl_handle * dl_load_library(ଅଚଳ fs::path & path);
ଶୂନ୍ୟ * dl_get_sym(dl_handle * handle, ଅଚଳ ଅକ୍ଷର * name);
ଅଚଳ ଅକ୍ଷର * dl_error();

