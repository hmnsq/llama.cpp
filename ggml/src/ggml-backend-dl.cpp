#ضمن "ggml-backend-dl.h"

#که_تعریف_شوی _WIN32

dl_handle * dl_بارول_library(ثابت fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_ډېرپېښ = سېټ_Error_ډېرپېښ(SEM_FAILCRITICALERRORS);
    سېټ_Error_ډېرپېښ(old_ډېرپېښ | SEM_FAILCRITICALERRORS);

    HMODULE handle = بارول_Library_W(path.wstring().c_str());

    سېټ_Error_ډېرپېښ(old_ډېرپېښ);

    أرجع handle;
}

فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    DWORD old_ډېرپېښ = سېټ_Error_ډېرپېښ(SEM_FAILCRITICALERRORS);
    سېټ_Error_ډېرپېښ(old_ډېرپېښ | SEM_FAILCRITICALERRORS);

    فراغ * p = (فراغ *) Get_Proc_پته(handle, name);

    سېټ_Error_ډېرپېښ(old_ډېرپېښ);

    أرجع p;
}

ثابت حرف * dl_error() {
    أرجع "";
}

#وإلا

dl_handle * dl_بارول_library(ثابت fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    أرجع handle;
}

فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    أرجع dlsym(handle, name);
}

ثابت حرف * dl_error() {
    ثابت حرف *rslt = dlerror();
    أرجع rslt != تش_اشاره ? rslt : "";
}

#د_که_پای
