#ଅନ୍ତର୍ଭାଵ "ggml-backend-dl.h"

#ifdef _WIN32

dl_handle * dl_load_library(ଅଚଳ fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    ଫେରନ୍ତୁ handle;
}

ଶୂନ୍ୟ * dl_get_sym(dl_handle * handle, ଅଚଳ ଅକ୍ଷର * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    ଶୂନ୍ୟ * p = (ଶୂନ୍ୟ *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    ଫେରନ୍ତୁ p;
}

ଅଚଳ ଅକ୍ଷର * dl_error() {
    ଫେରନ୍ତୁ "";
}

#ଅନ୍ୟଥା

dl_handle * dl_load_library(ଅଚଳ fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    ଫେରନ୍ତୁ handle;
}

ଶୂନ୍ୟ * dl_get_sym(dl_handle * handle, ଅଚଳ ଅକ୍ଷର * name) {
    ଫେରନ୍ତୁ dlsym(handle, name);
}

ଅଚଳ ଅକ୍ଷର * dl_error() {
    ଅଚଳ ଅକ୍ଷର *rslt = dlerror();
    ଫେରନ୍ତୁ rslt != nullptr ? rslt : "";
}

#endif
