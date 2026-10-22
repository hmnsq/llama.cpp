#અન્તર્ભાવ "ggml-backend-dl.h"

#ifdef _WIN32

dl_handle * dl_load_library(અચલ fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    પાછા handle;
}

શૂન્ય * dl_get_sym(dl_handle * handle, અચલ અક્ષર * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    શૂન્ય * p = (શૂન્ય *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    પાછા p;
}

અચલ અક્ષર * dl_error() {
    પાછા "";
}

#નહીંતર

dl_handle * dl_load_library(અચલ fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    પાછા handle;
}

શૂન્ય * dl_get_sym(dl_handle * handle, અચલ અક્ષર * name) {
    પાછા dlsym(handle, name);
}

અચલ અક્ષર * dl_error() {
    અચલ અક્ષર *rslt = dlerror();
    પાછા rslt != nullptr ? rslt : "";
}

#endif
