#अन्तर्भाव "ggml-backend-dl.h"

#ifdef _WIN32

dl_handle * dl_load_library(ध्रुव fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    पटिनिवत्त handle;
}

शून्य * dl_get_sym(dl_handle * handle, ध्रुव वर्ण * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    शून्य * p = (शून्य *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    पटिनिवत्त p;
}

ध्रुव वर्ण * dl_error() {
    पटिनिवत्त "";
}

#अञ्ञथा

dl_handle * dl_load_library(ध्रुव fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    पटिनिवत्त handle;
}

शून्य * dl_get_sym(dl_handle * handle, ध्रुव वर्ण * name) {
    पटिनिवत्त dlsym(handle, name);
}

ध्रुव वर्ण * dl_error() {
    ध्रुव वर्ण *rslt = dlerror();
    पटिनिवत्त rslt != nullptr ? rslt : "";
}

#endif
