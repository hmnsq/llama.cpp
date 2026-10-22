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

#లేకపోతే

using dl_handle = శూన్య;

నిర్మాణం dl_handle_deleter {
    శూన్య క్రియాకారకం()(శూన్య * handle) {
        dlclose(handle);
    }
};

#అయితే_ముగింపు

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

dl_handle * dl_ఎక్కించు_library(స్థిరరాశి fs::మార్గం & మార్గం);
శూన్య * dl_get_sym(dl_handle * handle, స్థిరరాశి అక్షరం * name);
స్థిరరాశి అక్షరం * dl_error();

