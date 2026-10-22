#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-dl.h"

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32

dl_handle * dl_ਲੱਦੋ_library(ਅਚਲ fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_ਬਹੁਲਕ = ਸੈੱਟ_Error_ਬਹੁਲਕ(SEM_FAILCRITICALERRORS);
    ਸੈੱਟ_Error_ਬਹੁਲਕ(old_ਬਹੁਲਕ | SEM_FAILCRITICALERRORS);

    HMODULE handle = ਲੱਦੋ_Library_W(path.wstring().c_str());

    ਸੈੱਟ_Error_ਬਹੁਲਕ(old_ਬਹੁਲਕ);

    ਵਾਪਸ handle;
}

ਖਾਲੀ * dl_get_sym(dl_handle * handle, ਅਚਲ ਅੱਖਰ * name) {
    DWORD old_ਬਹੁਲਕ = ਸੈੱਟ_Error_ਬਹੁਲਕ(SEM_FAILCRITICALERRORS);
    ਸੈੱਟ_Error_ਬਹੁਲਕ(old_ਬਹੁਲਕ | SEM_FAILCRITICALERRORS);

    ਖਾਲੀ * p = (ਖਾਲੀ *) Get_Proc_ਪਤਾ(handle, name);

    ਸੈੱਟ_Error_ਬਹੁਲਕ(old_ਬਹੁਲਕ);

    ਵਾਪਸ p;
}

ਅਚਲ ਅੱਖਰ * dl_error() {
    ਵਾਪਸ "";
}

#ਨਹੀਂਤਾਂ

dl_handle * dl_ਲੱਦੋ_library(ਅਚਲ fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    ਵਾਪਸ handle;
}

ਖਾਲੀ * dl_get_sym(dl_handle * handle, ਅਚਲ ਅੱਖਰ * name) {
    ਵਾਪਸ dlsym(handle, name);
}

ਅਚਲ ਅੱਖਰ * dl_error() {
    ਅਚਲ ਅੱਖਰ *rslt = dlerror();
    ਵਾਪਸ rslt != ਖਾਲੀ_ਸੂਚਕ ? rslt : "";
}

#ਜੇ_ਅੰਤ
