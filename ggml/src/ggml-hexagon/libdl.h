#निर्देशः once

#यदि_परिभाषितम् _WIN32
#   define WIN32_LEAN_AND_माध्यम्
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#अन्यथा
#    include <dlfcn.h>
#    include <unistd.h>
#यदि_अन्तः
#अन्तर्भाव <filesystem>

namespace fs = std::filesystem;

#यदि_परिभाषितम् _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

संरचना dl_handle_deleter {
    शून्य संक्रियकः()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

स्थिर अन्तःस्थ dl_handle * dl_आरोपय_library(ध्रुव fs::मार्गः & मार्गः) {
    // suppress error dialogs for missing DLLs
    DWORD old_बहुलकम् = समुच्चयः_Error_बहुलकम्(SEM_FAILCRITICALERRORS);
    समुच्चयः_Error_बहुलकम्(old_बहुलकम् | SEM_FAILCRITICALERRORS);

    HMODULE handle = आरोपय_Library_W(मार्गः.wstring().c_str());

    समुच्चयः_Error_बहुलकम्(old_बहुलकम्);

    प्रत्यागम handle;
}

स्थिर अन्तःस्थ शून्य * dl_get_sym(dl_handle * handle, ध्रुव वर्ण * name) {
    DWORD old_बहुलकम् = समुच्चयः_Error_बहुलकम्(SEM_FAILCRITICALERRORS);
    समुच्चयः_Error_बहुलकम्(old_बहुलकम् | SEM_FAILCRITICALERRORS);

    शून्य * p = (शून्य *) Get_Proc_स्थानाङ्कः(handle, name);

    समुच्चयः_Error_बहुलकम्(old_बहुलकम्);

    प्रत्यागम p;
}

स्थिर अन्तःस्थ ध्रुव वर्ण * dl_error() {
    प्रत्यागम "";
}

#अन्यथा

using dl_handle = शून्य;

संरचना dl_handle_deleter {
    शून्य संक्रियकः()(शून्य * handle) {
        dlclose(handle);
    }
};

स्थिर अन्तःस्थ dl_handle * dl_आरोपय_library(ध्रुव fs::मार्गः & मार्गः) {
    dl_handle * handle = dlopen(मार्गः.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    प्रत्यागम handle;
}

स्थिर अन्तःस्थ शून्य * dl_get_sym(dl_handle * handle, ध्रुव वर्ण * name) {
    प्रत्यागम dlsym(handle, name);
}

स्थिर अन्तःस्थ ध्रुव वर्ण * dl_error() {
    ध्रुव वर्ण *rslt = dlerror();
    प्रत्यागम rslt != nullptr ? rslt : "";
}

#यदि_अन्तः
