#شامل "ggml-backend-dl.h"

#اگر_تعریف_شدہ _WIN32

dl_handle * dl_بھرو_library(ثابت fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_منوال = سیٹ_Error_منوال(SEM_FAILCRITICALERRORS);
    سیٹ_Error_منوال(old_منوال | SEM_FAILCRITICALERRORS);

    HMODULE handle = بھرو_Library_W(path.wstring().c_str());

    سیٹ_Error_منوال(old_منوال);

    واپس handle;
}

خالی * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    DWORD old_منوال = سیٹ_Error_منوال(SEM_FAILCRITICALERRORS);
    سیٹ_Error_منوال(old_منوال | SEM_FAILCRITICALERRORS);

    خالی * p = (خالی *) Get_Proc_پتہ(handle, name);

    سیٹ_Error_منوال(old_منوال);

    واپس p;
}

ثابت حرف * dl_error() {
    واپس "";
}

#ورنہ

dl_handle * dl_بھرو_library(ثابت fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    واپس handle;
}

خالی * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    واپس dlsym(handle, name);
}

ثابت حرف * dl_error() {
    ثابت حرف *rslt = dlerror();
    واپس rslt != خالی_اشاریہ ? rslt : "";
}

#اگر_ختم
