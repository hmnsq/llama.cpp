#ضمن "ggml-backend-dl.h"

#ifdef _WIN32

dl_handle * dl_load_library(ثابت fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    إرجاع handle;
}

فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    فراغ * p = (فراغ *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    إرجاع p;
}

ثابت حرف * dl_error() {
    إرجاع "";
}

#وإلا

dl_handle * dl_load_library(ثابت fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    إرجاع handle;
}

فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    إرجاع dlsym(handle, name);
}

ثابت حرف * dl_error() {
    ثابت حرف *rslt = dlerror();
    إرجاع rslt != nullptr ? rslt : "";
}

#endif
