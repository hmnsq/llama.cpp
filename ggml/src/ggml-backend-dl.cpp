#অন্তর্ভাব "ggml-backend-dl.h"

#ifdef _WIN32

dl_handle * dl_load_library(ধ্রুব fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    প্রত্যাগম handle;
}

শূন্য * dl_get_sym(dl_handle * handle, ধ্রুব বর্ণ * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    শূন্য * p = (শূন্য *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    প্রত্যাগম p;
}

ধ্রুব বর্ণ * dl_error() {
    প্রত্যাগম "";
}

#অন্যথা

dl_handle * dl_load_library(ধ্রুব fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    প্রত্যাগম handle;
}

শূন্য * dl_get_sym(dl_handle * handle, ধ্রুব বর্ণ * name) {
    প্রত্যাগম dlsym(handle, name);
}

ধ্রুব বর্ণ * dl_error() {
    ধ্রুব বর্ণ *rslt = dlerror();
    প্রত্যাগম rslt != nullptr ? rslt : "";
}

#endif
