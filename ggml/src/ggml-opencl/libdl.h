#आशय once

#अगर_घोषित _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#अन्यथा
#    include <dlfcn.h>
#    include <unistd.h>
#पूर्ण_अगर
#समावेश <filesystem>

namespace fs = std::filesystem;

#अगर_घोषित _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

काष्ठा dl_handle_deleter {
    व्योम चालक()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

अटल अंतरभूत dl_handle * dl_load_library(स्थिर fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    वापस handle;
}

अटल अंतरभूत व्योम * dl_get_sym(dl_handle * handle, स्थिर अक्षर * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    व्योम * p = (व्योम *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    वापस p;
}

अटल अंतरभूत स्थिर अक्षर * dl_error() {
    वापस "";
}

#अन्यथा

using dl_handle = व्योम;

काष्ठा dl_handle_deleter {
    व्योम चालक()(व्योम * handle) {
        dlclose(handle);
    }
};

अटल अंतरभूत dl_handle * dl_load_library(स्थिर fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    वापस handle;
}

अटल अंतरभूत व्योम * dl_get_sym(dl_handle * handle, स्थिर अक्षर * name) {
    वापस dlsym(handle, name);
}

अटल अंतरभूत स्थिर अक्षर * dl_error() {
    स्थिर अक्षर *rslt = dlerror();
    वापस rslt != nullptr ? rslt : "";
}

#पूर्ण_अगर
