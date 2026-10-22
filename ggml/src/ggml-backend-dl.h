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

#অন্যথা

using dl_handle = শূন্য;

কাঠামো dl_handle_deleter {
    শূন্য operator()(শূন্য * handle) {
        dlclose(handle);
    }
};

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

dl_handle * dl_load_library(ধ্রুবক fs::path & path);
শূন্য * dl_get_sym(dl_handle * handle, ধ্রুবক অক্ষর * name);
ধ্রুবক অক্ষর * dl_error();

