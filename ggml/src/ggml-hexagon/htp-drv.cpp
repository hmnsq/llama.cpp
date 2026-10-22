#అన్తర్భావ <filesystem>
#అన్తర్భావ <set>
#అన్తర్భావ <sstream>
#అన్తర్భావ <string>

#నిర్వచితమైతే _WIN32
#   define WIN32_LEAN_AND_సగటు
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#లేకపోతే
#   include <dlfcn.h>
#   include <unistd.h>
#అయితే_ముగింపు

#సూచన clang diagnostic ignored "-Wgnu-anonymous-struct"
#సూచన clang diagnostic ignored "-Wmissing-prototypes"
#సూచన clang diagnostic ignored "-Wsign-compare"
#సూచన clang diagnostic ignored "-Wlanguage-extension-token"
#సూచన clang diagnostic ignored "-Wmicrosoft-enum-value"
#సూచన clang diagnostic ignored "-Wnested-anon-types"

#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "htp-drv.h"
#అన్తర్భావ "libdl.h"

#అన్తర్భావ <domain.h>

//
// Driver API types
//

రకనిర్వచనం శూన్య * (*rpcmem_alloc_pfn_t)(పూర్ణాంకం heapid, uint32_t flags, పూర్ణాంకం size);
రకనిర్వచనం శూన్య * (*rpcmem_alloc2_pfn_t)(పూర్ణాంకం heapid, uint32_t flags, size_t size);
రకనిర్వచనం శూన్య   (*rpcmem_free_pfn_t)(శూన్య * po);
రకనిర్వచనం పూర్ణాంకం    (*rpcmem_గమ్యం_fd_pfn_t)(శూన్య * po);

రకనిర్వచనం AEEResult (*dspqueue_create_pfn_t)(పూర్ణాంకం                 domain,
                                           uint32_t            flags,
                                           uint32_t            req_వరుస_కొలత,
                                           uint32_t            resp_వరుస_కొలత,
                                           dspqueue_callback_t packet_callback,
                                           dspqueue_callback_t error_callback,
                                           శూన్య *              callback_context,
                                           dspqueue_t *        queue);
రకనిర్వచనం AEEResult (*dspqueue_close_pfn_t)(dspqueue_t queue);
రకనిర్వచనం AEEResult (*dspqueue_export_pfn_t)(dspqueue_t queue, uint64_t *వరుస_id);
రకనిర్వచనం AEEResult (*dspqueue_write_pfn_t)(dspqueue_t queue, uint32_t flags,
                                          uint32_t అంకె_buffers,
                                          నిర్మాణం dspqueue_బఫరు *buffers,
                                          uint32_t సందేశం_నిడివి,
                                          స్థిరరాశి uint8_t *సందేశం,
                                          uint32_t timeout_us);
రకనిర్వచనం AEEResult (*dspqueue_read_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                         uint32_t max_buffers, uint32_t *అంకె_buffers,
                                         నిర్మాణం dspqueue_బఫరు *buffers,
                                         uint32_t max_సందేశం_నిడివి,
                                         uint32_t *సందేశం_నిడివి, uint8_t *సందేశం,
                                         uint32_t timeout_us);
రకనిర్వచనం AEEResult (*dspqueue_read_noblock_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                                 uint32_t max_buffers, uint32_t *అంకె_buffers,
                                                 నిర్మాణం dspqueue_బఫరు *buffers,
                                                 uint32_t max_సందేశం_నిడివి,
                                                 uint32_t *సందేశం_నిడివి, uint8_t *సందేశం);
రకనిర్వచనం పూర్ణాంకం (*fastrpc_mmap_pfn_t)(పూర్ణాంకం domain, పూర్ణాంకం fd, శూన్య *addr, పూర్ణాంకం స్థానభ్రంశం, size_t నిడివి, గణన fastrpc_map_flags flags);
రకనిర్వచనం పూర్ణాంకం (*fastrpc_munmap_pfn_t)(పూర్ణాంకం domain, పూర్ణాంకం fd, శూన్య *addr, size_t నిడివి);

రకనిర్వచనం పూర్ణాంకం (*remote_handle64_open_pfn_t)(స్థిరరాశి అక్షరం* name, remote_handle64 *ph);
రకనిర్వచనం పూర్ణాంకం (*remote_handle64_invoke_pfn_t)(remote_handle64 h, uint32_t dwScalars, remote_arg *pra);
రకనిర్వచనం పూర్ణాంకం (*remote_handle64_close_pfn_t)(remote_handle h);
రకనిర్వచనం పూర్ణాంకం (*remote_handle_control_pfn_t)(uint32_t req, శూన్య* దత్తాంశం, uint32_t datalen);
రకనిర్వచనం పూర్ణాంకం (*remote_handle64_control_pfn_t)(remote_handle64 h, uint32_t req, శూన్య* దత్తాంశం, uint32_t datalen);
రకనిర్వచనం పూర్ణాంకం (*remote_session_control_pfn_t)(uint32_t req, శూన్య *దత్తాంశం, uint32_t datalen);
రకనిర్వచనం పూర్ణాంకం (*remote_system_request_pfn_t)(system_req_payload * req);

//
// Driver API pfns
//

rpcmem_alloc_pfn_t  rpcmem_alloc_pfn  = శూన్యసూచిక;
rpcmem_alloc2_pfn_t rpcmem_alloc2_pfn = శూన్యసూచిక;
rpcmem_free_pfn_t   rpcmem_free_pfn   = శూన్యసూచిక;
rpcmem_గమ్యం_fd_pfn_t  rpcmem_గమ్యం_fd_pfn  = శూన్యసూచిక;

fastrpc_mmap_pfn_t   fastrpc_mmap_pfn   = శూన్యసూచిక;
fastrpc_munmap_pfn_t fastrpc_munmap_pfn = శూన్యసూచిక;

dspqueue_create_pfn_t       dspqueue_create_pfn       = శూన్యసూచిక;
dspqueue_close_pfn_t        dspqueue_close_pfn        = శూన్యసూచిక;
dspqueue_export_pfn_t       dspqueue_export_pfn       = శూన్యసూచిక;
dspqueue_write_pfn_t        dspqueue_write_pfn        = శూన్యసూచిక;
dspqueue_read_pfn_t         dspqueue_read_pfn         = శూన్యసూచిక;
dspqueue_read_noblock_pfn_t dspqueue_read_noblock_pfn = శూన్యసూచిక;

remote_handle64_open_pfn_t    remote_handle64_open_pfn    = శూన్యసూచిక;
remote_handle64_invoke_pfn_t  remote_handle64_invoke_pfn  = శూన్యసూచిక;
remote_handle64_close_pfn_t   remote_handle64_close_pfn   = శూన్యసూచిక;
remote_handle_control_pfn_t   remote_handle_control_pfn   = శూన్యసూచిక;
remote_handle64_control_pfn_t remote_handle64_control_pfn = శూన్యసూచిక;
remote_session_control_pfn_t  remote_session_control_pfn  = శూన్యసూచిక;
remote_system_request_pfn_t   remote_system_request_pfn   = శూన్యసూచిక;

//
// Driver API
//

శూన్య * rpcmem_alloc(పూర్ణాంకం heapid, uint32_t flags, పూర్ణాంకం size) {
    తిరిగి rpcmem_alloc_pfn(heapid, flags, size);
}

శూన్య * rpcmem_alloc2(పూర్ణాంకం heapid, uint32_t flags, size_t size) {
    ఒకవేళ (rpcmem_alloc2_pfn) {
        తిరిగి rpcmem_alloc2_pfn(heapid, flags, size);
    } లేకపోతే {
        GGML_LOG_INFO("ggml-hex: rpcmem_alloc2 not found, falling back to rpcmem_alloc\n");
        తిరిగి rpcmem_alloc_pfn(heapid, flags, size);
    }
}

శూన్య rpcmem_free(శూన్య * po) {
    తిరిగి rpcmem_free_pfn(po);
}

పూర్ణాంకం rpcmem_గమ్యం_fd(శూన్య * po) {
    తిరిగి rpcmem_గమ్యం_fd_pfn(po);
}

HTPDRV_API పూర్ణాంకం fastrpc_mmap(పూర్ణాంకం domain, పూర్ణాంకం fd, శూన్య * addr, పూర్ణాంకం స్థానభ్రంశం, size_t నిడివి, గణన fastrpc_map_flags flags) {
    తిరిగి fastrpc_mmap_pfn(domain, fd, addr, స్థానభ్రంశం, నిడివి, flags);
}

HTPDRV_API పూర్ణాంకం fastrpc_munmap(పూర్ణాంకం domain, పూర్ణాంకం fd, శూన్య * addr, size_t నిడివి) {
    తిరిగి fastrpc_munmap_pfn(domain, fd, addr, నిడివి);
}

AEEResult dspqueue_create(పూర్ణాంకం                 domain,
                          uint32_t            flags,
                          uint32_t            req_వరుస_కొలత,
                          uint32_t            resp_వరుస_కొలత,
                          dspqueue_callback_t packet_callback,
                          dspqueue_callback_t error_callback,
                          శూన్య *              callback_context,
                          dspqueue_t *        queue) {
    తిరిగి dspqueue_create_pfn(domain, flags, req_వరుస_కొలత, resp_వరుస_కొలత, packet_callback, error_callback,
                               callback_context, queue);
}

AEEResult dspqueue_close(dspqueue_t queue) {
    తిరిగి dspqueue_close_pfn(queue);
}

AEEResult dspqueue_export(dspqueue_t queue, uint64_t * వరుస_id) {
    తిరిగి dspqueue_export_pfn(queue, వరుస_id);
}

AEEResult dspqueue_write(dspqueue_t               queue,
                         uint32_t                 flags,
                         uint32_t                 అంకె_buffers,
                         నిర్మాణం dspqueue_బఫరు * buffers,
                         uint32_t                 సందేశం_నిడివి,
                         స్థిరరాశి uint8_t *          సందేశం,
                         uint32_t                 timeout_us) {
    తిరిగి dspqueue_write_pfn(queue, flags, అంకె_buffers, buffers, సందేశం_నిడివి, సందేశం, timeout_us);
}

AEEResult dspqueue_read(dspqueue_t               queue,
                        uint32_t *               flags,
                        uint32_t                 max_buffers,
                        uint32_t *               అంకె_buffers,
                        నిర్మాణం dspqueue_బఫరు * buffers,
                        uint32_t                 max_సందేశం_నిడివి,
                        uint32_t *               సందేశం_నిడివి,
                        uint8_t *                సందేశం,
                        uint32_t                 timeout_us) {
#నిర్వచితమైతే _WIN32
    ఒకవేళ (timeout_us == 0) {
        తిరిగి dspqueue_read_noblock_pfn(queue, flags, max_buffers, అంకె_buffers, buffers, max_సందేశం_నిడివి,
                                     సందేశం_నిడివి, సందేశం);
    }
#అయితే_ముగింపు
    తిరిగి dspqueue_read_pfn(queue, flags, max_buffers, అంకె_buffers, buffers, max_సందేశం_నిడివి, సందేశం_నిడివి,
                             సందేశం, timeout_us);
}

HTPDRV_API పూర్ణాంకం remote_handle64_open(స్థిరరాశి అక్షరం * name, remote_handle64 * ph) {
    తిరిగి remote_handle64_open_pfn(name, ph);
}

HTPDRV_API పూర్ణాంకం remote_handle64_invoke(remote_handle64 h, uint32_t dwScalars, remote_arg * pra) {
    తిరిగి remote_handle64_invoke_pfn(h, dwScalars, pra);
}

HTPDRV_API పూర్ణాంకం remote_handle64_close(remote_handle64 h) {
    తిరిగి remote_handle64_close_pfn(h);
}

HTPDRV_API పూర్ణాంకం remote_handle_control(uint32_t req, శూన్య * దత్తాంశం, uint32_t datalen) {
    తిరిగి remote_handle_control_pfn(req, దత్తాంశం, datalen);
}

HTPDRV_API పూర్ణాంకం remote_handle64_control(remote_handle64 h, uint32_t req, శూన్య * దత్తాంశం, uint32_t datalen) {
    తిరిగి remote_handle64_control_pfn(h, req, దత్తాంశం, datalen);
}

HTPDRV_API పూర్ణాంకం remote_session_control(uint32_t req, శూన్య * దత్తాంశం, uint32_t datalen) {
    తిరిగి remote_session_control_pfn(req, దత్తాంశం, datalen);
}

HTPDRV_API పూర్ణాంకం remote_system_request(system_req_payload * req) {
    ఒకవేళ (!remote_system_request_pfn) {
        తిరిగి AEE_EUNSUPPORTEDAPI;
    }
    తిరిగి remote_system_request_pfn(req);
}

#నిర్వచితమైతే _WIN32

స్థిర std::string wstr_గమ్యం_str(std::wstring_view wstr) {
    std::string ఫలితం;
    ఒకవేళ (wstr.ఖాళీలు()) {
        తిరిగి ఫలితం;
    }
    స్వయంచాలక బైట్లు_needed = Wide_Char_గమ్యం_Multi_Byte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.దత్తాంశం(), (పూర్ణాంకం) wstr.size(),
                                            శూన్యసూచిక, 0, శూన్యసూచిక, శూన్యసూచిక);
    ఒకవేళ (బైట్లు_needed == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", Get_చివరి_Error());
        విసురు std::runtime_error("Invalid wstring input");
    }

    ఫలితం.resize(బైట్లు_needed, '\0');
    పూర్ణాంకం బైట్లు_written = Wide_Char_గమ్యం_Multi_Byte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.దత్తాంశం(), (పూర్ణాంకం) wstr.size(),
                                            ఫలితం.దత్తాంశం(), బైట్లు_needed,
                                            శూన్యసూచిక, శూన్యసూచిక);
    ఒకవేళ (బైట్లు_written == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", Get_చివరి_Error());
        విసురు std::runtime_error("Wstring conversion failed");
    }
    తిరిగి ఫలితం;
}

స్థిర std::string get_driver_మార్గం() {
    std::wstring serviceName = L"qcnspmcdm";
    std::string ఫలితం;

    // Get a handle to the SCM database.
    SC_HANDLE schSCManager = OpenSCManagerW(ఖాళీ, ఖాళీ, STANDARD_RIGHTS_READ);
    ఒకవేళ (శూన్యసూచిక == schSCManager) {
        GGML_LOG_ERROR("ggml-hex: Failed to open SCManager. Error: %lu\n", Get_చివరి_Error());
        తిరిగి ఫలితం;
    }

    // Get a handle to the service.
    SC_HANDLE schService = OpenServiceW(schSCManager,           // SCM database
                                        serviceName.c_str(),    // name of service
                                        SERVICE_QUERY_CONFIG);  // need query config access

    ఒకవేళ (శూన్యసూచిక == schService) {
        GGML_LOG_ERROR("ggml-hex: Failed to open qcnspmcdm service. Error: %lu\n", Get_చివరి_Error());
        CloseServiceHandle(schSCManager);
        తిరిగి ఫలితం;
    }

    // Store the size of buffer used as an output.
    DWORD బఫరు_కొలత_2;
    ఒకవేళ (!QueryServiceConfigW(schService, ఖాళీ, 0, &బఫరు_కొలత_2) &&
        (Get_చివరి_Error() != ERROR_INSUFFICIENT_బఫరు)) {
        GGML_LOG_ERROR("ggml-hex: Failed to query service config. Error: %lu\n", Get_చివరి_Error());
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        తిరిగి ఫలితం;
    }
    // Get the configuration of the service.
    LPQUERY_SERVICE_CONFIGW serviceConfig =
        static_cast<LPQUERY_SERVICE_CONFIGW>(LocalAlloc(LMEM_FIXED, బఫరు_కొలత_2));
    ఒకవేళ (!QueryServiceConfigW(schService, serviceConfig, బఫరు_కొలత_2, &బఫరు_కొలత_2)) {
        fprintf(stderr, "ggml-hex: Failed to query service config. Error: %lu\n", Get_చివరి_Error());
        LocalFree(serviceConfig);
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        తిరిగి ఫలితం;
    }

    // Read the driver file path get its parent directory
    std::wstring driver_మార్గం = std::wstring(serviceConfig->lp_ద్విమాన_మార్గం_Name);
    driver_మార్గం = driver_మార్గం.substr(0, driver_మార్గం.find_last_of(L"\\"));

    // Clean up resources
    LocalFree(serviceConfig);
    CloseServiceHandle(schService);
    CloseServiceHandle(schSCManager);

    // Driver path would contain invalid path string, like:
    // \SystemRoot\System32\DriverStore\FileRepository\qcadsprpc8280.inf_arm64_c2b9460c9a072f37
    // "\SystemRoot" should be replace with a correct one (e.g. C:\Windows)
    స్థిరరాశి std::wstring system_మూలం_Placeholder = L"\\SystemRoot";
    ఒకవేళ (0 != driver_మార్గం.compare(0, system_మూలం_Placeholder.నిడివి(), system_మూలం_Placeholder)) {
        GGML_LOG_ERROR("ggml-hex: String pattern not found in driver path.\n");
        తిరిగి ఫలితం;
    }

    // Replace \SystemRoot with an absolute path from system ENV windir
    స్థిరరాశి std::wstring system_మూలం_Env = L"windir";

    // Query the number of wide characters this variable requires
    DWORD అంకె_పదాలు = GetEnvironmentVariableW(system_మూలం_Env.c_str(), ఖాళీ, 0);
    ఒకవేళ (అంకె_పదాలు == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed get systemRoot environment variable\n");
        తిరిగి ఫలితం;
    }

    // Query the actual system root name from environment variable
    std::vector<wchar_t> system_మూలం(అంకె_పదాలు + 1);
    అంకె_పదాలు = GetEnvironmentVariableW(system_మూలం_Env.c_str(), system_మూలం.దత్తాంశం(), అంకె_పదాలు + 1);
    ఒకవేళ (అంకె_పదాలు == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed to read windir environment variable\n");
        తిరిగి ఫలితం;
    }
    driver_మార్గం.replace(0, system_మూలం_Placeholder.నిడివి(), std::wstring(system_మూలం.దత్తాంశం()));

    తిరిగి wstr_గమ్యం_str(driver_మార్గం);
}

#అయితే_ముగింపు

ఉపయోగించి dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

పూర్ణాంకం htpdrv_init() {
    స్థిర dl_handle_ptr lib_cdsp_rpc_handle = శూన్యసూచిక;
    స్థిర తార్కిక initialized = అబద్ధం;
#నిర్వచితమైతే _WIN32
    std::string drv_మార్గం = get_driver_మార్గం() + "\\" + "libcdsprpc.dll";
#లేకపోతే
    std::string drv_మార్గం = "libcdsprpc.so";
#అయితే_ముగింపు
    ఒకవేళ (initialized) {
        GGML_LOG_INFO("ggml-hex: Driver already loaded\n");
        తిరిగి AEE_SUCCESS;
    }
    GGML_LOG_INFO("ggml-hex: Loading driver %s\n", drv_మార్గం.c_str());

    fs::path path{ drv_మార్గం.c_str() };
    dl_handle_ptr handle { dl_ఎక్కించు_library(path) };
    ఒకవేళ (!handle) {
        GGML_LOG_ERROR("ggml-hex: failed to load %s: %s\n", path.u8string().c_str(), dl_error());
        తిరిగి AEE_EUNABLETOLOAD;
    }

#నిర్వచించు dlsym(drv, type, pfn, చిహ్నం, ignore)                               \
    చేయి {                                                                    \
        pfn = (type) dl_get_sym(drv, #symbol);                              \
        ఒకవేళ (!ignore && శూన్యసూచిక == pfn) {                                    \
            GGML_LOG_ERROR("ggml-hex: failed to dlsym %s\n", #symbol);      \
            తిరిగి AEE_EUNABLETOLOAD;                                       \
        }                                                                   \
    } అయితే (0)

    dlsym(handle.get(), rpcmem_alloc_pfn_t, rpcmem_alloc_pfn, rpcmem_alloc, అబద్ధం);
    dlsym(handle.get(), rpcmem_alloc2_pfn_t, rpcmem_alloc2_pfn, rpcmem_alloc2, నిజం);
    dlsym(handle.get(), rpcmem_free_pfn_t, rpcmem_free_pfn, rpcmem_free, అబద్ధం);
    dlsym(handle.get(), rpcmem_గమ్యం_fd_pfn_t, rpcmem_గమ్యం_fd_pfn, rpcmem_గమ్యం_fd, అబద్ధం);
    dlsym(handle.get(), fastrpc_mmap_pfn_t, fastrpc_mmap_pfn, fastrpc_mmap, అబద్ధం);
    dlsym(handle.get(), fastrpc_munmap_pfn_t, fastrpc_munmap_pfn, fastrpc_munmap, అబద్ధం);
    dlsym(handle.get(), dspqueue_create_pfn_t, dspqueue_create_pfn, dspqueue_create, అబద్ధం);
    dlsym(handle.get(), dspqueue_close_pfn_t, dspqueue_close_pfn, dspqueue_close, అబద్ధం);
    dlsym(handle.get(), dspqueue_export_pfn_t, dspqueue_export_pfn, dspqueue_export, అబద్ధం);
    dlsym(handle.get(), dspqueue_write_pfn_t, dspqueue_write_pfn, dspqueue_write, అబద్ధం);
    dlsym(handle.get(), dspqueue_read_pfn_t, dspqueue_read_pfn, dspqueue_read, అబద్ధం);
    dlsym(handle.get(), dspqueue_read_noblock_pfn_t, dspqueue_read_noblock_pfn, dspqueue_read_noblock, అబద్ధం);
    dlsym(handle.get(), remote_handle64_open_pfn_t, remote_handle64_open_pfn, remote_handle64_open, అబద్ధం);
    dlsym(handle.get(), remote_handle64_invoke_pfn_t, remote_handle64_invoke_pfn, remote_handle64_invoke, అబద్ధం);
    dlsym(handle.get(), remote_handle_control_pfn_t, remote_handle_control_pfn, remote_handle_control, అబద్ధం);
    dlsym(handle.get(), remote_handle64_control_pfn_t, remote_handle64_control_pfn, remote_handle64_control, అబద్ధం);
    dlsym(handle.get(), remote_session_control_pfn_t, remote_session_control_pfn, remote_session_control, అబద్ధం);
    dlsym(handle.get(), remote_handle64_close_pfn_t, remote_handle64_close_pfn, remote_handle64_close, అబద్ధం);
    dlsym(handle.get(), remote_system_request_pfn_t, remote_system_request_pfn, remote_system_request, నిజం);

    lib_cdsp_rpc_handle = std::move(handle);
    initialized         = నిజం;

    తిరిగి AEE_SUCCESS;
}

domain * htpdrv_get_domain(పూర్ణాంకం domain_id) {
    పూర్ణాంకం i    = 0;
    పూర్ణాంకం size = పరిమాణం(supported_domains) / పరిమాణం(domain);

    కొరకు (i = 0; i < size; i++) {
        ఒకవేళ (supported_domains[i].id == domain_id) {
            తిరిగి &supported_domains[i];
        }
    }

    తిరిగి ఖాళీ;
}

పూర్ణాంకం htpdrv_get_arch(పూర్ణాంకం domain, పూర్ణాంకం * arch) {
    ఒకవేళ (!remote_handle_control_pfn) {
        GGML_LOG_ERROR("ggml-hex: remote_handle_control is not supported on this device\n");
        తిరిగి AEE_EUNSUPPORTEDAPI;
    }

    నిర్మాణం remote_dsp_capability arch_ver;
    arch_ver.domain       = (uint32_t) domain;
    arch_ver.attribute_ID = ARCH_VER;
    arch_ver.capability   = (uint32_t) 0;

    పూర్ణాంకం err = remote_handle_control(DSPRPC_GET_DSP_INFO, &arch_ver, పరిమాణం(arch_ver));
    ఒకవేళ ((err & 0xff) == (AEE_EUNSUPPORTEDAPI & 0xff)) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability API is not supported on this device\n");
        తిరిగి AEE_EUNSUPPORTEDAPI;
    }

    ఒకవేళ (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability query failed (err %d)\n", err);
        తిరిగి err;
    }

    uint32_t val = arch_ver.capability & 0xff;
    *arch = (పూర్ణాంకం) ((val >> 4) * 10 + (val & 0x0f));
    తిరిగి 0;
}
