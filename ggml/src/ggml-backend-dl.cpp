#ضمن "ggml-backend-dl.h"

#جيڪڏهن_وصف_ٿيل _WIN32

dl_handle * dl_ڀريو_library(ثابت fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_گهڻائي = سيٽ_Error_گهڻائي(SEM_FAILCRITICALERRORS);
    سيٽ_Error_گهڻائي(old_گهڻائي | SEM_FAILCRITICALERRORS);

    HMODULE handle = ڀريو_Library_W(path.wstring().c_str());

    سيٽ_Error_گهڻائي(old_گهڻائي);

    أرجع handle;
}

فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    DWORD old_گهڻائي = سيٽ_Error_گهڻائي(SEM_FAILCRITICALERRORS);
    سيٽ_Error_گهڻائي(old_گهڻائي | SEM_FAILCRITICALERRORS);

    فراغ * p = (فراغ *) Get_Proc_پتو(handle, name);

    سيٽ_Error_گهڻائي(old_گهڻائي);

    أرجع p;
}

ثابت حرف * dl_error() {
    أرجع "";
}

#وإلا

dl_handle * dl_ڀريو_library(ثابت fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    أرجع handle;
}

فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    أرجع dlsym(handle, name);
}

ثابت حرف * dl_error() {
    ثابت حرف *rslt = dlerror();
    أرجع rslt != خالي_اشارو ? rslt : "";
}

#جيڪڏهن_پورو
