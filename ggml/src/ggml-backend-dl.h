#هدايت once

#جيڪڏهن_وصف_ٿيل _WIN32
#   define WIN32_LEAN_AND_سراسري
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#وإلا
#    include <dlfcn.h>
#    include <unistd.h>
#جيڪڏهن_پورو
#ضمن <filesystem>

namespace fs = std::filesystem;

#جيڪڏهن_وصف_ٿيل _WIN32

using dl_handle = std::remove_اشارو_t<HMODULE>;

بنية dl_handle_deleter {
    فراغ عامل()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

#وإلا

using dl_handle = فراغ;

بنية dl_handle_deleter {
    فراغ عامل()(فراغ * handle) {
        dlclose(handle);
    }
};

#جيڪڏهن_پورو

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

dl_handle * dl_ڀريو_library(ثابت fs::رستو & رستو);
فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name);
ثابت حرف * dl_error();

