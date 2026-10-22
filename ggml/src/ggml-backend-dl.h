#ہدایت once

#جے_تعریف_شدہ _WIN32
#   define WIN32_LEAN_AND_اوسط
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#وإلا
#    include <dlfcn.h>
#    include <unistd.h>
#جے_انت
#ضمن <filesystem>

namespace fs = std::filesystem;

#جے_تعریف_شدہ _WIN32

using dl_handle = std::remove_اشارہ_گر_t<HMODULE>;

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

#جے_انت

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

dl_handle * dl_لدو_library(ثابت fs::رستہ & رستہ);
فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name);
ثابت حرف * dl_error();

