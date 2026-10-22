#अन्तर्भाव "ggml-backend-dl.h"

#यदि_परिभाषितम् _WIN32

dl_handle * dl_आरोपय_library(ध्रुव fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_बहुलकम् = समुच्चयः_Error_बहुलकम्(SEM_FAILCRITICALERRORS);
    समुच्चयः_Error_बहुलकम्(old_बहुलकम् | SEM_FAILCRITICALERRORS);

    HMODULE handle = आरोपय_Library_W(path.wstring().c_str());

    समुच्चयः_Error_बहुलकम्(old_बहुलकम्);

    प्रत्यागम handle;
}

शून्य * dl_get_sym(dl_handle * handle, ध्रुव वर्ण * name) {
    DWORD old_बहुलकम् = समुच्चयः_Error_बहुलकम्(SEM_FAILCRITICALERRORS);
    समुच्चयः_Error_बहुलकम्(old_बहुलकम् | SEM_FAILCRITICALERRORS);

    शून्य * p = (शून्य *) Get_Proc_स्थानाङ्कः(handle, name);

    समुच्चयः_Error_बहुलकम्(old_बहुलकम्);

    प्रत्यागम p;
}

ध्रुव वर्ण * dl_error() {
    प्रत्यागम "";
}

#अन्यथा

dl_handle * dl_आरोपय_library(ध्रुव fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    प्रत्यागम handle;
}

शून्य * dl_get_sym(dl_handle * handle, ध्रुव वर्ण * name) {
    प्रत्यागम dlsym(handle, name);
}

ध्रुव वर्ण * dl_error() {
    ध्रुव वर्ण *rslt = dlerror();
    प्रत्यागम rslt != शून्यसूचकः ? rslt : "";
}

#यदि_अन्तः
