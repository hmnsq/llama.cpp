#ಅನ್ತರ್ಭಾವ "ggml-backend-dl.h"

#ifdef _WIN32

dl_handle * dl_load_library(ಧ್ರುವ fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    ಪ್ರತ್ಯಾಗಮ handle;
}

ಶೂನ್ಯ * dl_get_sym(dl_handle * handle, ಧ್ರುವ ವರ್ಣ * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    ಶೂನ್ಯ * p = (ಶೂನ್ಯ *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    ಪ್ರತ್ಯಾಗಮ p;
}

ಧ್ರುವ ವರ್ಣ * dl_error() {
    ಪ್ರತ್ಯಾಗಮ "";
}

#ಅನ್ಯಥಾ

dl_handle * dl_load_library(ಧ್ರುವ fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    ಪ್ರತ್ಯಾಗಮ handle;
}

ಶೂನ್ಯ * dl_get_sym(dl_handle * handle, ಧ್ರುವ ವರ್ಣ * name) {
    ಪ್ರತ್ಯಾಗಮ dlsym(handle, name);
}

ಧ್ರುವ ವರ್ಣ * dl_error() {
    ಧ್ರುವ ವರ್ಣ *rslt = dlerror();
    ಪ್ರತ್ಯಾಗಮ rslt != nullptr ? rslt : "";
}

#endif
