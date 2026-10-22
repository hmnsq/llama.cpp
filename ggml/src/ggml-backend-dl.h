#pragma once

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#இல்லையெனில்
#    include <dlfcn.h>
#    include <unistd.h>
#endif
#அந்தர்பாவ <filesystem>

namespace fs = std::filesystem;

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

கட்டமைப்பு dl_handle_deleter {
    வெறுமை operator()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

#இல்லையெனில்

using dl_handle = வெறுமை;

கட்டமைப்பு dl_handle_deleter {
    வெறுமை operator()(வெறுமை * handle) {
        dlclose(handle);
    }
};

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

dl_handle * dl_load_library(மாறிலி fs::path & path);
வெறுமை * dl_get_sym(dl_handle * handle, மாறிலி எழுத்து * name);
மாறிலி எழுத்து * dl_error();

