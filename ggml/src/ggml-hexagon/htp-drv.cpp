#ਅਨ੍ਤਰ੍ਭਾਵ <filesystem>
#ਅਨ੍ਤਰ੍ਭਾਵ <set>
#ਅਨ੍ਤਰ੍ਭਾਵ <sstream>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
#   define WIN32_LEAN_AND_ਔਸਤ
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#ਨਹੀਂਤਾਂ
#   include <dlfcn.h>
#   include <unistd.h>
#ਜੇ_ਅੰਤ

#ਹਦਾਇਤ clang diagnostic ignored "-Wgnu-anonymous-struct"
#ਹਦਾਇਤ clang diagnostic ignored "-Wmissing-prototypes"
#ਹਦਾਇਤ clang diagnostic ignored "-Wsign-compare"
#ਹਦਾਇਤ clang diagnostic ignored "-Wlanguage-extension-token"
#ਹਦਾਇਤ clang diagnostic ignored "-Wmicrosoft-enum-value"
#ਹਦਾਇਤ clang diagnostic ignored "-Wnested-anon-types"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-drv.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "libdl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <domain.h>

//
// Driver API types
//

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ * (*rpcmem_alloc_pfn_t)(ਪੂਰਨ heapid, uint32_t flags, ਪੂਰਨ size);
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ * (*rpcmem_alloc2_pfn_t)(ਪੂਰਨ heapid, uint32_t flags, size_t size);
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ   (*rpcmem_free_pfn_t)(ਖਾਲੀ * po);
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਪੂਰਨ    (*rpcmem_ਮੰਜ਼ਿਲ_fd_pfn_t)(ਖਾਲੀ * po);

ਕਿਸਮਪਰਿਭਾਸ਼ਾ AEEResult (*dspqueue_create_pfn_t)(ਪੂਰਨ                 domain,
                                           uint32_t            flags,
                                           uint32_t            req_ਕਤਾਰ_ਮਾਪ,
                                           uint32_t            resp_ਕਤਾਰ_ਮਾਪ,
                                           dspqueue_callback_t packet_callback,
                                           dspqueue_callback_t error_callback,
                                           ਖਾਲੀ *              callback_context,
                                           dspqueue_t *        queue);
ਕਿਸਮਪਰਿਭਾਸ਼ਾ AEEResult (*dspqueue_close_pfn_t)(dspqueue_t queue);
ਕਿਸਮਪਰਿਭਾਸ਼ਾ AEEResult (*dspqueue_export_pfn_t)(dspqueue_t queue, uint64_t *ਕਤਾਰ_id);
ਕਿਸਮਪਰਿਭਾਸ਼ਾ AEEResult (*dspqueue_write_pfn_t)(dspqueue_t queue, uint32_t flags,
                                          uint32_t ਅੰਕ_buffers,
                                          ਬਣਤਰ dspqueue_ਬਫਰ *buffers,
                                          uint32_t ਸੁਨੇਹਾ_ਲੰਮਾਈ,
                                          ਅਚਲ uint8_t *ਸੁਨੇਹਾ,
                                          uint32_t timeout_us);
ਕਿਸਮਪਰਿਭਾਸ਼ਾ AEEResult (*dspqueue_read_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                         uint32_t max_buffers, uint32_t *ਅੰਕ_buffers,
                                         ਬਣਤਰ dspqueue_ਬਫਰ *buffers,
                                         uint32_t max_ਸੁਨੇਹਾ_ਲੰਮਾਈ,
                                         uint32_t *ਸੁਨੇਹਾ_ਲੰਮਾਈ, uint8_t *ਸੁਨੇਹਾ,
                                         uint32_t timeout_us);
ਕਿਸਮਪਰਿਭਾਸ਼ਾ AEEResult (*dspqueue_read_noblock_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                                 uint32_t max_buffers, uint32_t *ਅੰਕ_buffers,
                                                 ਬਣਤਰ dspqueue_ਬਫਰ *buffers,
                                                 uint32_t max_ਸੁਨੇਹਾ_ਲੰਮਾਈ,
                                                 uint32_t *ਸੁਨੇਹਾ_ਲੰਮਾਈ, uint8_t *ਸੁਨੇਹਾ);
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਪੂਰਨ (*fastrpc_mmap_pfn_t)(ਪੂਰਨ domain, ਪੂਰਨ fd, ਖਾਲੀ *addr, ਪੂਰਨ ਵਿੱਥ, size_t ਲੰਮਾਈ, ਗਿਣਤੀ fastrpc_map_flags flags);
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਪੂਰਨ (*fastrpc_munmap_pfn_t)(ਪੂਰਨ domain, ਪੂਰਨ fd, ਖਾਲੀ *addr, size_t ਲੰਮਾਈ);

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਪੂਰਨ (*remote_handle64_open_pfn_t)(ਅਚਲ ਅੱਖਰ* name, remote_handle64 *ph);
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਪੂਰਨ (*remote_handle64_invoke_pfn_t)(remote_handle64 h, uint32_t dwScalars, remote_arg *pra);
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਪੂਰਨ (*remote_handle64_close_pfn_t)(remote_handle h);
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਪੂਰਨ (*remote_handle_control_pfn_t)(uint32_t req, ਖਾਲੀ* ਅੰਕੜੇ, uint32_t datalen);
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਪੂਰਨ (*remote_handle64_control_pfn_t)(remote_handle64 h, uint32_t req, ਖਾਲੀ* ਅੰਕੜੇ, uint32_t datalen);
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਪੂਰਨ (*remote_session_control_pfn_t)(uint32_t req, ਖਾਲੀ *ਅੰਕੜੇ, uint32_t datalen);
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਪੂਰਨ (*remote_system_request_pfn_t)(system_req_payload * req);

//
// Driver API pfns
//

rpcmem_alloc_pfn_t  rpcmem_alloc_pfn  = ਖਾਲੀ_ਸੂਚਕ;
rpcmem_alloc2_pfn_t rpcmem_alloc2_pfn = ਖਾਲੀ_ਸੂਚਕ;
rpcmem_free_pfn_t   rpcmem_free_pfn   = ਖਾਲੀ_ਸੂਚਕ;
rpcmem_ਮੰਜ਼ਿਲ_fd_pfn_t  rpcmem_ਮੰਜ਼ਿਲ_fd_pfn  = ਖਾਲੀ_ਸੂਚਕ;

fastrpc_mmap_pfn_t   fastrpc_mmap_pfn   = ਖਾਲੀ_ਸੂਚਕ;
fastrpc_munmap_pfn_t fastrpc_munmap_pfn = ਖਾਲੀ_ਸੂਚਕ;

dspqueue_create_pfn_t       dspqueue_create_pfn       = ਖਾਲੀ_ਸੂਚਕ;
dspqueue_close_pfn_t        dspqueue_close_pfn        = ਖਾਲੀ_ਸੂਚਕ;
dspqueue_export_pfn_t       dspqueue_export_pfn       = ਖਾਲੀ_ਸੂਚਕ;
dspqueue_write_pfn_t        dspqueue_write_pfn        = ਖਾਲੀ_ਸੂਚਕ;
dspqueue_read_pfn_t         dspqueue_read_pfn         = ਖਾਲੀ_ਸੂਚਕ;
dspqueue_read_noblock_pfn_t dspqueue_read_noblock_pfn = ਖਾਲੀ_ਸੂਚਕ;

remote_handle64_open_pfn_t    remote_handle64_open_pfn    = ਖਾਲੀ_ਸੂਚਕ;
remote_handle64_invoke_pfn_t  remote_handle64_invoke_pfn  = ਖਾਲੀ_ਸੂਚਕ;
remote_handle64_close_pfn_t   remote_handle64_close_pfn   = ਖਾਲੀ_ਸੂਚਕ;
remote_handle_control_pfn_t   remote_handle_control_pfn   = ਖਾਲੀ_ਸੂਚਕ;
remote_handle64_control_pfn_t remote_handle64_control_pfn = ਖਾਲੀ_ਸੂਚਕ;
remote_session_control_pfn_t  remote_session_control_pfn  = ਖਾਲੀ_ਸੂਚਕ;
remote_system_request_pfn_t   remote_system_request_pfn   = ਖਾਲੀ_ਸੂਚਕ;

//
// Driver API
//

ਖਾਲੀ * rpcmem_alloc(ਪੂਰਨ heapid, uint32_t flags, ਪੂਰਨ size) {
    ਵਾਪਸ rpcmem_alloc_pfn(heapid, flags, size);
}

ਖਾਲੀ * rpcmem_alloc2(ਪੂਰਨ heapid, uint32_t flags, size_t size) {
    ਜੇ (rpcmem_alloc2_pfn) {
        ਵਾਪਸ rpcmem_alloc2_pfn(heapid, flags, size);
    } ਨਹੀਂਤਾਂ {
        GGML_LOG_INFO("ggml-hex: rpcmem_alloc2 not found, falling back to rpcmem_alloc\n");
        ਵਾਪਸ rpcmem_alloc_pfn(heapid, flags, size);
    }
}

ਖਾਲੀ rpcmem_free(ਖਾਲੀ * po) {
    ਵਾਪਸ rpcmem_free_pfn(po);
}

ਪੂਰਨ rpcmem_ਮੰਜ਼ਿਲ_fd(ਖਾਲੀ * po) {
    ਵਾਪਸ rpcmem_ਮੰਜ਼ਿਲ_fd_pfn(po);
}

HTPDRV_API ਪੂਰਨ fastrpc_mmap(ਪੂਰਨ domain, ਪੂਰਨ fd, ਖਾਲੀ * addr, ਪੂਰਨ ਵਿੱਥ, size_t ਲੰਮਾਈ, ਗਿਣਤੀ fastrpc_map_flags flags) {
    ਵਾਪਸ fastrpc_mmap_pfn(domain, fd, addr, ਵਿੱਥ, ਲੰਮਾਈ, flags);
}

HTPDRV_API ਪੂਰਨ fastrpc_munmap(ਪੂਰਨ domain, ਪੂਰਨ fd, ਖਾਲੀ * addr, size_t ਲੰਮਾਈ) {
    ਵਾਪਸ fastrpc_munmap_pfn(domain, fd, addr, ਲੰਮਾਈ);
}

AEEResult dspqueue_create(ਪੂਰਨ                 domain,
                          uint32_t            flags,
                          uint32_t            req_ਕਤਾਰ_ਮਾਪ,
                          uint32_t            resp_ਕਤਾਰ_ਮਾਪ,
                          dspqueue_callback_t packet_callback,
                          dspqueue_callback_t error_callback,
                          ਖਾਲੀ *              callback_context,
                          dspqueue_t *        queue) {
    ਵਾਪਸ dspqueue_create_pfn(domain, flags, req_ਕਤਾਰ_ਮਾਪ, resp_ਕਤਾਰ_ਮਾਪ, packet_callback, error_callback,
                               callback_context, queue);
}

AEEResult dspqueue_close(dspqueue_t queue) {
    ਵਾਪਸ dspqueue_close_pfn(queue);
}

AEEResult dspqueue_export(dspqueue_t queue, uint64_t * ਕਤਾਰ_id) {
    ਵਾਪਸ dspqueue_export_pfn(queue, ਕਤਾਰ_id);
}

AEEResult dspqueue_write(dspqueue_t               queue,
                         uint32_t                 flags,
                         uint32_t                 ਅੰਕ_buffers,
                         ਬਣਤਰ dspqueue_ਬਫਰ * buffers,
                         uint32_t                 ਸੁਨੇਹਾ_ਲੰਮਾਈ,
                         ਅਚਲ uint8_t *          ਸੁਨੇਹਾ,
                         uint32_t                 timeout_us) {
    ਵਾਪਸ dspqueue_write_pfn(queue, flags, ਅੰਕ_buffers, buffers, ਸੁਨੇਹਾ_ਲੰਮਾਈ, ਸੁਨੇਹਾ, timeout_us);
}

AEEResult dspqueue_read(dspqueue_t               queue,
                        uint32_t *               flags,
                        uint32_t                 max_buffers,
                        uint32_t *               ਅੰਕ_buffers,
                        ਬਣਤਰ dspqueue_ਬਫਰ * buffers,
                        uint32_t                 max_ਸੁਨੇਹਾ_ਲੰਮਾਈ,
                        uint32_t *               ਸੁਨੇਹਾ_ਲੰਮਾਈ,
                        uint8_t *                ਸੁਨੇਹਾ,
                        uint32_t                 timeout_us) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
    ਜੇ (timeout_us == 0) {
        ਵਾਪਸ dspqueue_read_noblock_pfn(queue, flags, max_buffers, ਅੰਕ_buffers, buffers, max_ਸੁਨੇਹਾ_ਲੰਮਾਈ,
                                     ਸੁਨੇਹਾ_ਲੰਮਾਈ, ਸੁਨੇਹਾ);
    }
#ਜੇ_ਅੰਤ
    ਵਾਪਸ dspqueue_read_pfn(queue, flags, max_buffers, ਅੰਕ_buffers, buffers, max_ਸੁਨੇਹਾ_ਲੰਮਾਈ, ਸੁਨੇਹਾ_ਲੰਮਾਈ,
                             ਸੁਨੇਹਾ, timeout_us);
}

HTPDRV_API ਪੂਰਨ remote_handle64_open(ਅਚਲ ਅੱਖਰ * name, remote_handle64 * ph) {
    ਵਾਪਸ remote_handle64_open_pfn(name, ph);
}

HTPDRV_API ਪੂਰਨ remote_handle64_invoke(remote_handle64 h, uint32_t dwScalars, remote_arg * pra) {
    ਵਾਪਸ remote_handle64_invoke_pfn(h, dwScalars, pra);
}

HTPDRV_API ਪੂਰਨ remote_handle64_close(remote_handle64 h) {
    ਵਾਪਸ remote_handle64_close_pfn(h);
}

HTPDRV_API ਪੂਰਨ remote_handle_control(uint32_t req, ਖਾਲੀ * ਅੰਕੜੇ, uint32_t datalen) {
    ਵਾਪਸ remote_handle_control_pfn(req, ਅੰਕੜੇ, datalen);
}

HTPDRV_API ਪੂਰਨ remote_handle64_control(remote_handle64 h, uint32_t req, ਖਾਲੀ * ਅੰਕੜੇ, uint32_t datalen) {
    ਵਾਪਸ remote_handle64_control_pfn(h, req, ਅੰਕੜੇ, datalen);
}

HTPDRV_API ਪੂਰਨ remote_session_control(uint32_t req, ਖਾਲੀ * ਅੰਕੜੇ, uint32_t datalen) {
    ਵਾਪਸ remote_session_control_pfn(req, ਅੰਕੜੇ, datalen);
}

HTPDRV_API ਪੂਰਨ remote_system_request(system_req_payload * req) {
    ਜੇ (!remote_system_request_pfn) {
        ਵਾਪਸ AEE_EUNSUPPORTEDAPI;
    }
    ਵਾਪਸ remote_system_request_pfn(req);
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32

ਸਥਿਰ std::string wstr_ਮੰਜ਼ਿਲ_str(std::wstring_view wstr) {
    std::string ਨਤੀਜਾ;
    ਜੇ (wstr.ਖਾਲੀ_ਥਾਵਾਂ()) {
        ਵਾਪਸ ਨਤੀਜਾ;
    }
    ਸਵੈਚਾਲਿਤ ਬਾਈਟਾਂ_needed = Wide_Char_ਮੰਜ਼ਿਲ_Multi_Byte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.ਅੰਕੜੇ(), (ਪੂਰਨ) wstr.size(),
                                            ਖਾਲੀ_ਸੂਚਕ, 0, ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ);
    ਜੇ (ਬਾਈਟਾਂ_needed == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", Get_ਆਖਰੀ_Error());
        ਸੁੱਟੋ std::runtime_error("Invalid wstring input");
    }

    ਨਤੀਜਾ.resize(ਬਾਈਟਾਂ_needed, '\0');
    ਪੂਰਨ ਬਾਈਟਾਂ_written = Wide_Char_ਮੰਜ਼ਿਲ_Multi_Byte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.ਅੰਕੜੇ(), (ਪੂਰਨ) wstr.size(),
                                            ਨਤੀਜਾ.ਅੰਕੜੇ(), ਬਾਈਟਾਂ_needed,
                                            ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ);
    ਜੇ (ਬਾਈਟਾਂ_written == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", Get_ਆਖਰੀ_Error());
        ਸੁੱਟੋ std::runtime_error("Wstring conversion failed");
    }
    ਵਾਪਸ ਨਤੀਜਾ;
}

ਸਥਿਰ std::string get_driver_ਰਸਤਾ() {
    std::wstring serviceName = L"qcnspmcdm";
    std::string ਨਤੀਜਾ;

    // Get a handle to the SCM database.
    SC_HANDLE schSCManager = OpenSCManagerW(ਨਿਰਾ, ਨਿਰਾ, STANDARD_RIGHTS_READ);
    ਜੇ (ਖਾਲੀ_ਸੂਚਕ == schSCManager) {
        GGML_LOG_ERROR("ggml-hex: Failed to open SCManager. Error: %lu\n", Get_ਆਖਰੀ_Error());
        ਵਾਪਸ ਨਤੀਜਾ;
    }

    // Get a handle to the service.
    SC_HANDLE schService = OpenServiceW(schSCManager,           // SCM database
                                        serviceName.c_str(),    // name of service
                                        SERVICE_QUERY_CONFIG);  // need query config access

    ਜੇ (ਖਾਲੀ_ਸੂਚਕ == schService) {
        GGML_LOG_ERROR("ggml-hex: Failed to open qcnspmcdm service. Error: %lu\n", Get_ਆਖਰੀ_Error());
        CloseServiceHandle(schSCManager);
        ਵਾਪਸ ਨਤੀਜਾ;
    }

    // Store the size of buffer used as an output.
    DWORD ਬਫਰ_ਮਾਪ_2;
    ਜੇ (!QueryServiceConfigW(schService, ਨਿਰਾ, 0, &ਬਫਰ_ਮਾਪ_2) &&
        (Get_ਆਖਰੀ_Error() != ERROR_INSUFFICIENT_ਬਫਰ)) {
        GGML_LOG_ERROR("ggml-hex: Failed to query service config. Error: %lu\n", Get_ਆਖਰੀ_Error());
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        ਵਾਪਸ ਨਤੀਜਾ;
    }
    // Get the configuration of the service.
    LPQUERY_SERVICE_CONFIGW serviceConfig =
        static_cast<LPQUERY_SERVICE_CONFIGW>(LocalAlloc(LMEM_FIXED, ਬਫਰ_ਮਾਪ_2));
    ਜੇ (!QueryServiceConfigW(schService, serviceConfig, ਬਫਰ_ਮਾਪ_2, &ਬਫਰ_ਮਾਪ_2)) {
        fprintf(stderr, "ggml-hex: Failed to query service config. Error: %lu\n", Get_ਆਖਰੀ_Error());
        LocalFree(serviceConfig);
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        ਵਾਪਸ ਨਤੀਜਾ;
    }

    // Read the driver file path get its parent directory
    std::wstring driver_ਰਸਤਾ = std::wstring(serviceConfig->lp_ਦੁਆਧਾਰੀ_ਰਸਤਾ_Name);
    driver_ਰਸਤਾ = driver_ਰਸਤਾ.substr(0, driver_ਰਸਤਾ.find_last_of(L"\\"));

    // Clean up resources
    LocalFree(serviceConfig);
    CloseServiceHandle(schService);
    CloseServiceHandle(schSCManager);

    // Driver path would contain invalid path string, like:
    // \SystemRoot\System32\DriverStore\FileRepository\qcadsprpc8280.inf_arm64_c2b9460c9a072f37
    // "\SystemRoot" should be replace with a correct one (e.g. C:\Windows)
    ਅਚਲ std::wstring system_ਜੜ੍ਹ_Placeholder = L"\\SystemRoot";
    ਜੇ (0 != driver_ਰਸਤਾ.compare(0, system_ਜੜ੍ਹ_Placeholder.ਲੰਮਾਈ(), system_ਜੜ੍ਹ_Placeholder)) {
        GGML_LOG_ERROR("ggml-hex: String pattern not found in driver path.\n");
        ਵਾਪਸ ਨਤੀਜਾ;
    }

    // Replace \SystemRoot with an absolute path from system ENV windir
    ਅਚਲ std::wstring system_ਜੜ੍ਹ_Env = L"windir";

    // Query the number of wide characters this variable requires
    DWORD ਅੰਕ_ਸ਼ਬਦ = GetEnvironmentVariableW(system_ਜੜ੍ਹ_Env.c_str(), ਨਿਰਾ, 0);
    ਜੇ (ਅੰਕ_ਸ਼ਬਦ == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed get systemRoot environment variable\n");
        ਵਾਪਸ ਨਤੀਜਾ;
    }

    // Query the actual system root name from environment variable
    std::vector<wchar_t> system_ਜੜ੍ਹ(ਅੰਕ_ਸ਼ਬਦ + 1);
    ਅੰਕ_ਸ਼ਬਦ = GetEnvironmentVariableW(system_ਜੜ੍ਹ_Env.c_str(), system_ਜੜ੍ਹ.ਅੰਕੜੇ(), ਅੰਕ_ਸ਼ਬਦ + 1);
    ਜੇ (ਅੰਕ_ਸ਼ਬਦ == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed to read windir environment variable\n");
        ਵਾਪਸ ਨਤੀਜਾ;
    }
    driver_ਰਸਤਾ.replace(0, system_ਜੜ੍ਹ_Placeholder.ਲੰਮਾਈ(), std::wstring(system_ਜੜ੍ਹ.ਅੰਕੜੇ()));

    ਵਾਪਸ wstr_ਮੰਜ਼ਿਲ_str(driver_ਰਸਤਾ);
}

#ਜੇ_ਅੰਤ

ਵਰਤੋਂ dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

ਪੂਰਨ htpdrv_init() {
    ਸਥਿਰ dl_handle_ptr lib_cdsp_rpc_handle = ਖਾਲੀ_ਸੂਚਕ;
    ਸਥਿਰ ਤਾਰਕਿਕ initialized = ਝੂਠ;
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
    std::string drv_ਰਸਤਾ = get_driver_ਰਸਤਾ() + "\\" + "libcdsprpc.dll";
#ਨਹੀਂਤਾਂ
    std::string drv_ਰਸਤਾ = "libcdsprpc.so";
#ਜੇ_ਅੰਤ
    ਜੇ (initialized) {
        GGML_LOG_INFO("ggml-hex: Driver already loaded\n");
        ਵਾਪਸ AEE_SUCCESS;
    }
    GGML_LOG_INFO("ggml-hex: Loading driver %s\n", drv_ਰਸਤਾ.c_str());

    fs::path path{ drv_ਰਸਤਾ.c_str() };
    dl_handle_ptr handle { dl_ਲੱਦੋ_library(path) };
    ਜੇ (!handle) {
        GGML_LOG_ERROR("ggml-hex: failed to load %s: %s\n", path.u8string().c_str(), dl_error());
        ਵਾਪਸ AEE_EUNABLETOLOAD;
    }

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ dlsym(drv, type, pfn, ਚਿੰਨ੍ਹ, ignore)                               \
    ਕਰੋ {                                                                    \
        pfn = (type) dl_get_sym(drv, #symbol);                              \
        ਜੇ (!ignore && ਖਾਲੀ_ਸੂਚਕ == pfn) {                                    \
            GGML_LOG_ERROR("ggml-hex: failed to dlsym %s\n", #symbol);      \
            ਵਾਪਸ AEE_EUNABLETOLOAD;                                       \
        }                                                                   \
    } ਜਦੋਂ (0)

    dlsym(handle.get(), rpcmem_alloc_pfn_t, rpcmem_alloc_pfn, rpcmem_alloc, ਝੂਠ);
    dlsym(handle.get(), rpcmem_alloc2_pfn_t, rpcmem_alloc2_pfn, rpcmem_alloc2, ਸੱਚ);
    dlsym(handle.get(), rpcmem_free_pfn_t, rpcmem_free_pfn, rpcmem_free, ਝੂਠ);
    dlsym(handle.get(), rpcmem_ਮੰਜ਼ਿਲ_fd_pfn_t, rpcmem_ਮੰਜ਼ਿਲ_fd_pfn, rpcmem_ਮੰਜ਼ਿਲ_fd, ਝੂਠ);
    dlsym(handle.get(), fastrpc_mmap_pfn_t, fastrpc_mmap_pfn, fastrpc_mmap, ਝੂਠ);
    dlsym(handle.get(), fastrpc_munmap_pfn_t, fastrpc_munmap_pfn, fastrpc_munmap, ਝੂਠ);
    dlsym(handle.get(), dspqueue_create_pfn_t, dspqueue_create_pfn, dspqueue_create, ਝੂਠ);
    dlsym(handle.get(), dspqueue_close_pfn_t, dspqueue_close_pfn, dspqueue_close, ਝੂਠ);
    dlsym(handle.get(), dspqueue_export_pfn_t, dspqueue_export_pfn, dspqueue_export, ਝੂਠ);
    dlsym(handle.get(), dspqueue_write_pfn_t, dspqueue_write_pfn, dspqueue_write, ਝੂਠ);
    dlsym(handle.get(), dspqueue_read_pfn_t, dspqueue_read_pfn, dspqueue_read, ਝੂਠ);
    dlsym(handle.get(), dspqueue_read_noblock_pfn_t, dspqueue_read_noblock_pfn, dspqueue_read_noblock, ਝੂਠ);
    dlsym(handle.get(), remote_handle64_open_pfn_t, remote_handle64_open_pfn, remote_handle64_open, ਝੂਠ);
    dlsym(handle.get(), remote_handle64_invoke_pfn_t, remote_handle64_invoke_pfn, remote_handle64_invoke, ਝੂਠ);
    dlsym(handle.get(), remote_handle_control_pfn_t, remote_handle_control_pfn, remote_handle_control, ਝੂਠ);
    dlsym(handle.get(), remote_handle64_control_pfn_t, remote_handle64_control_pfn, remote_handle64_control, ਝੂਠ);
    dlsym(handle.get(), remote_session_control_pfn_t, remote_session_control_pfn, remote_session_control, ਝੂਠ);
    dlsym(handle.get(), remote_handle64_close_pfn_t, remote_handle64_close_pfn, remote_handle64_close, ਝੂਠ);
    dlsym(handle.get(), remote_system_request_pfn_t, remote_system_request_pfn, remote_system_request, ਸੱਚ);

    lib_cdsp_rpc_handle = std::move(handle);
    initialized         = ਸੱਚ;

    ਵਾਪਸ AEE_SUCCESS;
}

domain * htpdrv_get_domain(ਪੂਰਨ domain_id) {
    ਪੂਰਨ i    = 0;
    ਪੂਰਨ size = ਆਕਾਰ(supported_domains) / ਆਕਾਰ(domain);

    ਲਈ (i = 0; i < size; i++) {
        ਜੇ (supported_domains[i].id == domain_id) {
            ਵਾਪਸ &supported_domains[i];
        }
    }

    ਵਾਪਸ ਨਿਰਾ;
}

ਪੂਰਨ htpdrv_get_arch(ਪੂਰਨ domain, ਪੂਰਨ * arch) {
    ਜੇ (!remote_handle_control_pfn) {
        GGML_LOG_ERROR("ggml-hex: remote_handle_control is not supported on this device\n");
        ਵਾਪਸ AEE_EUNSUPPORTEDAPI;
    }

    ਬਣਤਰ remote_dsp_capability arch_ver;
    arch_ver.domain       = (uint32_t) domain;
    arch_ver.attribute_ID = ARCH_VER;
    arch_ver.capability   = (uint32_t) 0;

    ਪੂਰਨ err = remote_handle_control(DSPRPC_GET_DSP_INFO, &arch_ver, ਆਕਾਰ(arch_ver));
    ਜੇ ((err & 0xff) == (AEE_EUNSUPPORTEDAPI & 0xff)) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability API is not supported on this device\n");
        ਵਾਪਸ AEE_EUNSUPPORTEDAPI;
    }

    ਜੇ (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability query failed (err %d)\n", err);
        ਵਾਪਸ err;
    }

    uint32_t val = arch_ver.capability & 0xff;
    *arch = (ਪੂਰਨ) ((val >> 4) * 10 + (val & 0x0f));
    ਵਾਪਸ 0;
}
