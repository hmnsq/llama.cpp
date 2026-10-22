#அந்தர்பாவ "ggml-backend-dl.h"

#ifdef _WIN32

dl_handle * dl_load_library(மாறிலி fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    திரும்பு handle;
}

வெறுமை * dl_get_sym(dl_handle * handle, மாறிலி எழுத்து * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    வெறுமை * p = (வெறுமை *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    திரும்பு p;
}

மாறிலி எழுத்து * dl_error() {
    திரும்பு "";
}

#இல்லையெனில்

dl_handle * dl_load_library(மாறிலி fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    திரும்பு handle;
}

வெறுமை * dl_get_sym(dl_handle * handle, மாறிலி எழுத்து * name) {
    திரும்பு dlsym(handle, name);
}

மாறிலி எழுத்து * dl_error() {
    மாறிலி எழுத்து *rslt = dlerror();
    திரும்பு rslt != nullptr ? rslt : "";
}

#endif
