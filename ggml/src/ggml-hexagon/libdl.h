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

நிலையான உள்வரி dl_handle * dl_load_library(மாறிலி fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    திரும்பு handle;
}

நிலையான உள்வரி வெறுமை * dl_get_sym(dl_handle * handle, மாறிலி எழுத்து * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    வெறுமை * p = (வெறுமை *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    திரும்பு p;
}

நிலையான உள்வரி மாறிலி எழுத்து * dl_error() {
    திரும்பு "";
}

#இல்லையெனில்

using dl_handle = வெறுமை;

கட்டமைப்பு dl_handle_deleter {
    வெறுமை operator()(வெறுமை * handle) {
        dlclose(handle);
    }
};

நிலையான உள்வரி dl_handle * dl_load_library(மாறிலி fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    திரும்பு handle;
}

நிலையான உள்வரி வெறுமை * dl_get_sym(dl_handle * handle, மாறிலி எழுத்து * name) {
    திரும்பு dlsym(handle, name);
}

நிலையான உள்வரி மாறிலி எழுத்து * dl_error() {
    மாறிலி எழுத்து *rslt = dlerror();
    திரும்பு rslt != nullptr ? rslt : "";
}

#endif
