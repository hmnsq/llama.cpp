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

#अन्यथा

using dl_handle = व्योम;

काष्ठा dl_handle_deleter {
    व्योम चालक()(व्योम * handle) {
        dlclose(handle);
    }
};

#पूर्ण_अगर

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

dl_handle * dl_load_library(स्थिर fs::path & path);
व्योम * dl_get_sym(dl_handle * handle, स्थिर अक्षर * name);
स्थिर अक्षर * dl_error();

