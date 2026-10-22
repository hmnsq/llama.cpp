#ضمن "ggml-backend-dl.h"

#اگر_تعریف_شده _WIN32

dl_handle * dl_بارگیری_library(ثابت fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_نما = مجموعه_Error_نما(SEM_FAILCRITICALERRORS);
    مجموعه_Error_نما(old_نما | SEM_FAILCRITICALERRORS);

    HMODULE handle = بارگیری_Library_W(path.wstring().c_str());

    مجموعه_Error_نما(old_نما);

    أرجع handle;
}

فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    DWORD old_نما = مجموعه_Error_نما(SEM_FAILCRITICALERRORS);
    مجموعه_Error_نما(old_نما | SEM_FAILCRITICALERRORS);

    فراغ * p = (فراغ *) Get_Proc_آدرس(handle, name);

    مجموعه_Error_نما(old_نما);

    أرجع p;
}

ثابت حرف * dl_error() {
    أرجع "";
}

#وإلا

dl_handle * dl_بارگیری_library(ثابت fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    أرجع handle;
}

فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    أرجع dlsym(handle, name);
}

ثابت حرف * dl_error() {
    ثابت حرف *rslt = dlerror();
    أرجع rslt != اشاره_تهی ? rslt : "";
}

#پایان_اگر
