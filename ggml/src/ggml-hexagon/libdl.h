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

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ dl_handle * dl_load_library(ಧ್ರುವ fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    ಪ್ರತ್ಯಾಗಮ handle;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ * dl_get_sym(dl_handle * handle, ಧ್ರುವ ವರ್ಣ * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    ಶೂನ್ಯ * p = (ಶೂನ್ಯ *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    ಪ್ರತ್ಯಾಗಮ p;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಧ್ರುವ ವರ್ಣ * dl_error() {
    ಪ್ರತ್ಯಾಗಮ "";
}

#ಅನ್ಯಥಾ

using dl_handle = ಶೂನ್ಯ;

ಸಂರಚನಾ dl_handle_deleter {
    ಶೂನ್ಯ operator()(ಶೂನ್ಯ * handle) {
        dlclose(handle);
    }
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ dl_handle * dl_load_library(ಧ್ರುವ fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    ಪ್ರತ್ಯಾಗಮ handle;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ * dl_get_sym(dl_handle * handle, ಧ್ರುವ ವರ್ಣ * name) {
    ಪ್ರತ್ಯಾಗಮ dlsym(handle, name);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಧ್ರುವ ವರ್ಣ * dl_error() {
    ಧ್ರುವ ವರ್ಣ *rslt = dlerror();
    ಪ್ರತ್ಯಾಗಮ rslt != nullptr ? rslt : "";
}

#endif
