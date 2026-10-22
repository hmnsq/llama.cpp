#अन्तर्भाव "ggml-backend-dl.h"

#ifdef _WIN32

dl_handle * dl_load_library(स्थिरांक fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    परत handle;
}

शून्य * dl_get_sym(dl_handle * handle, स्थिरांक अक्षर * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    शून्य * p = (शून्य *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    परत p;
}

स्थिरांक अक्षर * dl_error() {
    परत "";
}

#अन्यथा

dl_handle * dl_load_library(स्थिरांक fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    परत handle;
}

शून्य * dl_get_sym(dl_handle * handle, स्थिरांक अक्षर * name) {
    परत dlsym(handle, name);
}

स्थिरांक अक्षर * dl_error() {
    स्थिरांक अक्षर *rslt = dlerror();
    परत rslt != nullptr ? rslt : "";
}

#endif
