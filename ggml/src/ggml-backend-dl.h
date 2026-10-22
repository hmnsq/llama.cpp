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

#ܐܚܪܢܐ

using dl_handle = فراغ;

بنية dl_handle_deleter {
    فراغ operator()(فراغ * handle) {
        dlclose(handle);
    }
};

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

dl_handle * dl_load_library(ثابت fs::path & path);
فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name);
ثابت حرف * dl_error();

