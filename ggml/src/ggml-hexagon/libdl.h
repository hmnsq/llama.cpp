#సూచన once

#నిర్వచితమైతే _WIN32
#   define WIN32_LEAN_AND_సగటు
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#లేకపోతే
#    include <dlfcn.h>
#    include <unistd.h>
#అయితే_ముగింపు
#అన్తర్భావ <filesystem>

namespace fs = std::filesystem;

#నిర్వచితమైతే _WIN32

using dl_handle = std::remove_సూచిక_t<HMODULE>;

నిర్మాణం dl_handle_deleter {
    శూన్య క్రియాకారకం()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

స్థిర అంతఃవరుస dl_handle * dl_ఎక్కించు_library(స్థిరరాశి fs::మార్గం & మార్గం) {
    // suppress error dialogs for missing DLLs
    DWORD old_బహుళకం = సమితి_Error_బహుళకం(SEM_FAILCRITICALERRORS);
    సమితి_Error_బహుళకం(old_బహుళకం | SEM_FAILCRITICALERRORS);

    HMODULE handle = ఎక్కించు_Library_W(మార్గం.wstring().c_str());

    సమితి_Error_బహుళకం(old_బహుళకం);

    తిరిగి handle;
}

స్థిర అంతఃవరుస శూన్య * dl_get_sym(dl_handle * handle, స్థిరరాశి అక్షరం * name) {
    DWORD old_బహుళకం = సమితి_Error_బహుళకం(SEM_FAILCRITICALERRORS);
    సమితి_Error_బహుళకం(old_బహుళకం | SEM_FAILCRITICALERRORS);

    శూన్య * p = (శూన్య *) Get_Proc_చిరునామా(handle, name);

    సమితి_Error_బహుళకం(old_బహుళకం);

    తిరిగి p;
}

స్థిర అంతఃవరుస స్థిరరాశి అక్షరం * dl_error() {
    తిరిగి "";
}

#లేకపోతే

using dl_handle = శూన్య;

నిర్మాణం dl_handle_deleter {
    శూన్య క్రియాకారకం()(శూన్య * handle) {
        dlclose(handle);
    }
};

స్థిర అంతఃవరుస dl_handle * dl_ఎక్కించు_library(స్థిరరాశి fs::మార్గం & మార్గం) {
    dl_handle * handle = dlopen(మార్గం.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    తిరిగి handle;
}

స్థిర అంతఃవరుస శూన్య * dl_get_sym(dl_handle * handle, స్థిరరాశి అక్షరం * name) {
    తిరిగి dlsym(handle, name);
}

స్థిర అంతఃవరుస స్థిరరాశి అక్షరం * dl_error() {
    స్థిరరాశి అక్షరం *rslt = dlerror();
    తిరిగి rslt != nullptr ? rslt : "";
}

#అయితే_ముగింపు
