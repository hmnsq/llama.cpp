#অন্তর্ভাৱ "ggml-backend-dl.h"

#ifdef _WIN32

dl_handle * dl_load_library(ধ্রুৱ fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    ঘূৰাই handle;
}

শূন্য * dl_get_sym(dl_handle * handle, ধ্রুৱ ৱর্ণ * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    শূন্য * p = (শূন্য *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    ঘূৰাই p;
}

ধ্রুৱ ৱর্ণ * dl_error() {
    ঘূৰাই "";
}

#অন্যথা

dl_handle * dl_load_library(ধ্রুৱ fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    ঘূৰাই handle;
}

শূন্য * dl_get_sym(dl_handle * handle, ধ্রুৱ ৱর্ণ * name) {
    ঘূৰাই dlsym(handle, name);
}

ধ্রুৱ ৱর্ণ * dl_error() {
    ধ্রুৱ ৱর্ণ *rslt = dlerror();
    ঘূৰাই rslt != nullptr ? rslt : "";
}

#endif
