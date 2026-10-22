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

ਸਥਿਰ ਸਤਰਅੰਦਰ dl_handle * dl_ਲੱਦੋ_library(ਅਚਲ fs::ਰਸਤਾ & ਰਸਤਾ) {
    // suppress error dialogs for missing DLLs
    DWORD old_ਬਹੁਲਕ = ਸੈੱਟ_Error_ਬਹੁਲਕ(SEM_FAILCRITICALERRORS);
    ਸੈੱਟ_Error_ਬਹੁਲਕ(old_ਬਹੁਲਕ | SEM_FAILCRITICALERRORS);

    HMODULE handle = ਲੱਦੋ_Library_W(ਰਸਤਾ.wstring().c_str());

    ਸੈੱਟ_Error_ਬਹੁਲਕ(old_ਬਹੁਲਕ);

    ਵਾਪਸ handle;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ * dl_get_sym(dl_handle * handle, ਅਚਲ ਅੱਖਰ * name) {
    DWORD old_ਬਹੁਲਕ = ਸੈੱਟ_Error_ਬਹੁਲਕ(SEM_FAILCRITICALERRORS);
    ਸੈੱਟ_Error_ਬਹੁਲਕ(old_ਬਹੁਲਕ | SEM_FAILCRITICALERRORS);

    ਖਾਲੀ * p = (ਖਾਲੀ *) Get_Proc_ਪਤਾ(handle, name);

    ਸੈੱਟ_Error_ਬਹੁਲਕ(old_ਬਹੁਲਕ);

    ਵਾਪਸ p;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਅਚਲ ਅੱਖਰ * dl_error() {
    ਵਾਪਸ "";
}

#ਨਹੀਂਤਾਂ

using dl_handle = ਖਾਲੀ;

ਬਣਤਰ dl_handle_deleter {
    ਖਾਲੀ ਸੰਚਾਲਕ()(ਖਾਲੀ * handle) {
        dlclose(handle);
    }
};

ਸਥਿਰ ਸਤਰਅੰਦਰ dl_handle * dl_ਲੱਦੋ_library(ਅਚਲ fs::ਰਸਤਾ & ਰਸਤਾ) {
    dl_handle * handle = dlopen(ਰਸਤਾ.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    ਵਾਪਸ handle;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ * dl_get_sym(dl_handle * handle, ਅਚਲ ਅੱਖਰ * name) {
    ਵਾਪਸ dlsym(handle, name);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਅਚਲ ਅੱਖਰ * dl_error() {
    ਅਚਲ ਅੱਖਰ *rslt = dlerror();
    ਵਾਪਸ rslt != nullptr ? rslt : "";
}

#ਜੇ_ਅੰਤ
