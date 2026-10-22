#pragma once

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#अञ्ञथा
#    include <dlfcn.h>
#    include <unistd.h>
#endif
#अन्तर्भाव <filesystem>

namespace fs = std::filesystem;

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

संरचना dl_handle_deleter {
    शून्य operator()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

स्थिर अन्तःस्थ dl_handle * dl_load_library(ध्रुव fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    पटिनिवत्त handle;
}

स्थिर अन्तःस्थ शून्य * dl_get_sym(dl_handle * handle, ध्रुव वर्ण * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    शून्य * p = (शून्य *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    पटिनिवत्त p;
}

स्थिर अन्तःस्थ ध्रुव वर्ण * dl_error() {
    पटिनिवत्त "";
}

#अञ्ञथा

using dl_handle = शून्य;

संरचना dl_handle_deleter {
    शून्य operator()(शून्य * handle) {
        dlclose(handle);
    }
};

स्थिर अन्तःस्थ dl_handle * dl_load_library(ध्रुव fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    पटिनिवत्त handle;
}

स्थिर अन्तःस्थ शून्य * dl_get_sym(dl_handle * handle, ध्रुव वर्ण * name) {
    पटिनिवत्त dlsym(handle, name);
}

स्थिर अन्तःस्थ ध्रुव वर्ण * dl_error() {
    ध्रुव वर्ण *rslt = dlerror();
    पटिनिवत्त rslt != nullptr ? rslt : "";
}

#endif
