#అన్తర్భావ "ggml-backend-dl.h"

#నిర్వచితమైతే _WIN32

dl_handle * dl_ఎక్కించు_library(స్థిరరాశి fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_బహుళకం = సమితి_Error_బహుళకం(SEM_FAILCRITICALERRORS);
    సమితి_Error_బహుళకం(old_బహుళకం | SEM_FAILCRITICALERRORS);

    HMODULE handle = ఎక్కించు_Library_W(path.wstring().c_str());

    సమితి_Error_బహుళకం(old_బహుళకం);

    తిరిగి handle;
}

శూన్య * dl_get_sym(dl_handle * handle, స్థిరరాశి అక్షరం * name) {
    DWORD old_బహుళకం = సమితి_Error_బహుళకం(SEM_FAILCRITICALERRORS);
    సమితి_Error_బహుళకం(old_బహుళకం | SEM_FAILCRITICALERRORS);

    శూన్య * p = (శూన్య *) Get_Proc_చిరునామా(handle, name);

    సమితి_Error_బహుళకం(old_బహుళకం);

    తిరిగి p;
}

స్థిరరాశి అక్షరం * dl_error() {
    తిరిగి "";
}

#లేకపోతే

dl_handle * dl_ఎక్కించు_library(స్థిరరాశి fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    తిరిగి handle;
}

శూన్య * dl_get_sym(dl_handle * handle, స్థిరరాశి అక్షరం * name) {
    తిరిగి dlsym(handle, name);
}

స్థిరరాశి అక్షరం * dl_error() {
    స్థిరరాశి అక్షరం *rslt = dlerror();
    తిరిగి rslt != శూన్యసూచిక ? rslt : "";
}

#అయితే_ముగింపు
