#अन्तर्भाव "ggml-backend-dl.h"

#ifdef _WIN32

dl_handle * dl_load_library(अचर fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    फर्काऊ handle;
}

शून्य * dl_get_sym(dl_handle * handle, अचर वर्ण * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    शून्य * p = (शून्य *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    फर्काऊ p;
}

अचर वर्ण * dl_error() {
    फर्काऊ "";
}

#अन्यथा

dl_handle * dl_load_library(अचर fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    फर्काऊ handle;
}

शून्य * dl_get_sym(dl_handle * handle, अचर वर्ण * name) {
    फर्काऊ dlsym(handle, name);
}

अचर वर्ण * dl_error() {
    अचर वर्ण *rslt = dlerror();
    फर्काऊ rslt != nullptr ? rslt : "";
}

#endif
