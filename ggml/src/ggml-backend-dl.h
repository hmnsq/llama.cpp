#ਹਦਾਇਤ once

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
#   define WIN32_LEAN_AND_ਔਸਤ
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#ਨਹੀਂਤਾਂ
#    include <dlfcn.h>
#    include <unistd.h>
#ਜੇ_ਅੰਤ
#ਅਨ੍ਤਰ੍ਭਾਵ <filesystem>

namespace fs = std::filesystem;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32

using dl_handle = std::remove_ਸੂਚਕ_t<HMODULE>;

ਬਣਤਰ dl_handle_deleter {
    ਖਾਲੀ ਸੰਚਾਲਕ()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

#ਨਹੀਂਤਾਂ

using dl_handle = ਖਾਲੀ;

ਬਣਤਰ dl_handle_deleter {
    ਖਾਲੀ ਸੰਚਾਲਕ()(ਖਾਲੀ * handle) {
        dlclose(handle);
    }
};

#ਜੇ_ਅੰਤ

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

dl_handle * dl_ਲੱਦੋ_library(ਅਚਲ fs::ਰਸਤਾ & ਰਸਤਾ);
ਖਾਲੀ * dl_get_sym(dl_handle * handle, ਅਚਲ ਅੱਖਰ * name);
ਅਚਲ ਅੱਖਰ * dl_error();

