#pragma once

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#નહીંતર
#    include <dlfcn.h>
#    include <unistd.h>
#endif
#અન્તર્ભાવ <filesystem>

namespace fs = std::filesystem;

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

સંરચના dl_handle_deleter {
    શૂન્ય operator()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

#નહીંતર

using dl_handle = શૂન્ય;

સંરચના dl_handle_deleter {
    શૂન્ય operator()(શૂન્ય * handle) {
        dlclose(handle);
    }
};

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

dl_handle * dl_load_library(અચલ fs::path & path);
શૂન્ય * dl_get_sym(dl_handle * handle, અચલ અક્ષર * name);
અચલ અક્ષર * dl_error();

