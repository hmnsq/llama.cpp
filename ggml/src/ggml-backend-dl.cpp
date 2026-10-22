#അന്തര്ഭാവ "ggml-backend-dl.h"

#ifdef _WIN32

dl_handle * dl_load_library(സ്ഥിരം fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    തിരികെ handle;
}

ശൂന്യം * dl_get_sym(dl_handle * handle, സ്ഥിരം അക്ഷരം * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    ശൂന്യം * p = (ശൂന്യം *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    തിരികെ p;
}

സ്ഥിരം അക്ഷരം * dl_error() {
    തിരികെ "";
}

#അല്ലെങ്കിൽ

dl_handle * dl_load_library(സ്ഥിരം fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    തിരികെ handle;
}

ശൂന്യം * dl_get_sym(dl_handle * handle, സ്ഥിരം അക്ഷരം * name) {
    തിരികെ dlsym(handle, name);
}

സ്ഥിരം അക്ഷരം * dl_error() {
    സ്ഥിരം അക്ഷരം *rslt = dlerror();
    തിരികെ rslt != nullptr ? rslt : "";
}

#endif
