#অন্তর্ভাব "ggml-backend-dl.h"

#ifdef _WIN32

dl_handle * dl_load_library(ধ্রুবক fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    ফেরত handle;
}

শূন্য * dl_get_sym(dl_handle * handle, ধ্রুবক অক্ষর * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    শূন্য * p = (শূন্য *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    ফেরত p;
}

ধ্রুবক অক্ষর * dl_error() {
    ফেরত "";
}

#অন্যথা

dl_handle * dl_load_library(ধ্রুবক fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    ফেরত handle;
}

শূন্য * dl_get_sym(dl_handle * handle, ধ্রুবক অক্ষর * name) {
    ফেরত dlsym(handle, name);
}

ধ্রুবক অক্ষর * dl_error() {
    ধ্রুবক অক্ষর *rslt = dlerror();
    ফেরত rslt != nullptr ? rslt : "";
}

#endif
