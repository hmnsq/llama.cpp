#شامل "ggml-backend-dl.h"

#ifdef _WIN32

dl_handle * dl_load_library(ثابت fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    بازگشت handle;
}

تهی * dl_get_sym(dl_handle * handle, ثابت نویسه * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    تهی * p = (تهی *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    بازگشت p;
}

ثابت نویسه * dl_error() {
    بازگشت "";
}

#وگرنه

dl_handle * dl_load_library(ثابت fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    بازگشت handle;
}

تهی * dl_get_sym(dl_handle * handle, ثابت نویسه * name) {
    بازگشت dlsym(handle, name);
}

ثابت نویسه * dl_error() {
    ثابت نویسه *rslt = dlerror();
    بازگشت rslt != nullptr ? rslt : "";
}

#endif
