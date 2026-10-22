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

ସ୍ଥିର ଧାଡ଼ିଭିତରେ dl_handle * dl_load_library(ଅଚଳ fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    ଫେରନ୍ତୁ handle;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ * dl_get_sym(dl_handle * handle, ଅଚଳ ଅକ୍ଷର * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    ଶୂନ୍ୟ * p = (ଶୂନ୍ୟ *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    ଫେରନ୍ତୁ p;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଅଚଳ ଅକ୍ଷର * dl_error() {
    ଫେରନ୍ତୁ "";
}

#ଅନ୍ୟଥା

using dl_handle = ଶୂନ୍ୟ;

ସଂରଚନା dl_handle_deleter {
    ଶୂନ୍ୟ operator()(ଶୂନ୍ୟ * handle) {
        dlclose(handle);
    }
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ dl_handle * dl_load_library(ଅଚଳ fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    ଫେରନ୍ତୁ handle;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ * dl_get_sym(dl_handle * handle, ଅଚଳ ଅକ୍ଷର * name) {
    ଫେରନ୍ତୁ dlsym(handle, name);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଅଚଳ ଅକ୍ଷର * dl_error() {
    ଅଚଳ ଅକ୍ଷର *rslt = dlerror();
    ଫେରନ୍ତୁ rslt != nullptr ? rslt : "";
}

#endif
