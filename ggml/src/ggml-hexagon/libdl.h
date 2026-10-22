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

स्थिर अन्तःस्थ dl_handle * dl_load_library(अचर fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    फर्काऊ handle;
}

स्थिर अन्तःस्थ शून्य * dl_get_sym(dl_handle * handle, अचर वर्ण * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    शून्य * p = (शून्य *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    फर्काऊ p;
}

स्थिर अन्तःस्थ अचर वर्ण * dl_error() {
    फर्काऊ "";
}

#अन्यथा

using dl_handle = शून्य;

संरचना dl_handle_deleter {
    शून्य चालक()(शून्य * handle) {
        dlclose(handle);
    }
};

स्थिर अन्तःस्थ dl_handle * dl_load_library(अचर fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    फर्काऊ handle;
}

स्थिर अन्तःस्थ शून्य * dl_get_sym(dl_handle * handle, अचर वर्ण * name) {
    फर्काऊ dlsym(handle, name);
}

स्थिर अन्तःस्थ अचर वर्ण * dl_error() {
    अचर वर्ण *rslt = dlerror();
    फर्काऊ rslt != nullptr ? rslt : "";
}

#endif
