#pragma once

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#अण्णहा
#    include <dlfcn.h>
#    include <unistd.h>
#endif
#अन्तर्भाव <filesystem>

namespace fs = std::filesystem;

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

संरचना dl_handle_deleter {
    शून्य operator()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

#अण्णहा

using dl_handle = शून्य;

संरचना dl_handle_deleter {
    शून्य operator()(शून्य * handle) {
        dlclose(handle);
    }
};

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

dl_handle * dl_load_library(ध्रुव fs::path & path);
शून्य * dl_get_sym(dl_handle * handle, ध्रुव वर्ण * name);
ध्रुव वर्ण * dl_error();

