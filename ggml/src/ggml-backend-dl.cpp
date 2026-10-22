#ضمن "ggml-backend-dl.h"

#ifdef _WIN32

dl_handle * dl_load_library(ثابت fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    החזר handle;
}

فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    فراغ * p = (فراغ *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    החזר p;
}

ثابت حرف * dl_error() {
    החזר "";
}

#אחרת

dl_handle * dl_load_library(ثابت fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    החזר handle;
}

فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    החזר dlsym(handle, name);
}

ثابت حرف * dl_error() {
    ثابت حرف *rslt = dlerror();
    החזר rslt != nullptr ? rslt : "";
}

#endif
