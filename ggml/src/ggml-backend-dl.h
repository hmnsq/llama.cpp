#ہدایت once

#اگر_تعریف_شدہ _WIN32
#   define WIN32_LEAN_AND_اوسط
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#ورنہ
#    include <dlfcn.h>
#    include <unistd.h>
#اگر_ختم
#شامل <filesystem>

namespace fs = std::filesystem;

#اگر_تعریف_شدہ _WIN32

using dl_handle = std::remove_اشارہ_گر_t<HMODULE>;

ساخت dl_handle_deleter {
    خالی عامل()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

#ورنہ

using dl_handle = خالی;

ساخت dl_handle_deleter {
    خالی عامل()(خالی * handle) {
        dlclose(handle);
    }
};

#اگر_ختم

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

dl_handle * dl_بھرو_library(ثابت fs::راستہ & راستہ);
خالی * dl_get_sym(dl_handle * handle, ثابت حرف * name);
ثابت حرف * dl_error();

