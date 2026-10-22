#निर्देशः once

#यदि_परिभाषितम् _WIN32
#   define WIN32_LEAN_AND_माध्यम्
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#अन्यथा
#    include <dlfcn.h>
#    include <unistd.h>
#यदि_अन्तः
#अन्तर्भाव <filesystem>

namespace fs = std::filesystem;

#यदि_परिभाषितम् _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

संरचना dl_handle_deleter {
    शून्य संक्रियकः()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

#अन्यथा

using dl_handle = शून्य;

संरचना dl_handle_deleter {
    शून्य संक्रियकः()(शून्य * handle) {
        dlclose(handle);
    }
};

#यदि_अन्तः

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

dl_handle * dl_आरोपय_library(ध्रुव fs::मार्गः & मार्गः);
शून्य * dl_get_sym(dl_handle * handle, ध्रुव वर्ण * name);
ध्रुव वर्ण * dl_error();

