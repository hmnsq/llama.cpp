#अन्तर्भाव "ggml-backend-dl.h"

#ifdef _WIN32

dl_handle * dl_load_library(ध्रुव fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    प्रत्यागम handle;
}

शून्य * dl_get_sym(dl_handle * handle, ध्रुव वर्ण * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    शून्य * p = (शून्य *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    प्रत्यागम p;
}

ध्रुव वर्ण * dl_error() {
    प्रत्यागम "";
}

#अन्यथा

dl_handle * dl_load_library(ध्रुव fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    प्रत्यागम handle;
}

शून्य * dl_get_sym(dl_handle * handle, ध्रुव वर्ण * name) {
    प्रत्यागम dlsym(handle, name);
}

ध्रुव वर्ण * dl_error() {
    ध्रुव वर्ण *rslt = dlerror();
    प्रत्यागम rslt != nullptr ? rslt : "";
}

#endif
