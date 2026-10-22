#pragma once

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#अन्यथा
#    include <dlfcn.h>
#    include <unistd.h>
#endif
#अन्तर्भाव <filesystem>

namespace fs = std::filesystem;

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

संरचना dl_handle_deleter {
    शून्य चालक()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

स्थिर ओळीत dl_handle * dl_load_library(स्थिरांक fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    परत handle;
}

स्थिर ओळीत शून्य * dl_get_sym(dl_handle * handle, स्थिरांक अक्षर * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    शून्य * p = (शून्य *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    परत p;
}

स्थिर ओळीत स्थिरांक अक्षर * dl_error() {
    परत "";
}

#अन्यथा

using dl_handle = शून्य;

संरचना dl_handle_deleter {
    शून्य चालक()(शून्य * handle) {
        dlclose(handle);
    }
};

स्थिर ओळीत dl_handle * dl_load_library(स्थिरांक fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    परत handle;
}

स्थिर ओळीत शून्य * dl_get_sym(dl_handle * handle, स्थिरांक अक्षर * name) {
    परत dlsym(handle, name);
}

स्थिर ओळीत स्थिरांक अक्षर * dl_error() {
    स्थिरांक अक्षर *rslt = dlerror();
    परत rslt != nullptr ? rslt : "";
}

#endif
