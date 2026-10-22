#समावेश "ggml-backend-dl.h"

#अगर_घोषित _WIN32

dl_handle * dl_load_library(स्थिर fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    वापस handle;
}

व्योम * dl_get_sym(dl_handle * handle, स्थिर अक्षर * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    व्योम * p = (व्योम *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    वापस p;
}

स्थिर अक्षर * dl_error() {
    वापस "";
}

#अन्यथा

dl_handle * dl_load_library(स्थिर fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    वापस handle;
}

व्योम * dl_get_sym(dl_handle * handle, स्थिर अक्षर * name) {
    वापस dlsym(handle, name);
}

स्थिर अक्षर * dl_error() {
    स्थिर अक्षर *rslt = dlerror();
    वापस rslt != nullptr ? rslt : "";
}

#पूर्ण_अगर
