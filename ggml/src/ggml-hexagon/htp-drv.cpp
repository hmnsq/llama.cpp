#ಅನ್ತರ್ಭಾವ <filesystem>
#ಅನ್ತರ್ಭಾವ <set>
#ಅನ್ತರ್ಭಾವ <sstream>
#ಅನ್ತರ್ಭಾವ <string>

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
#   define WIN32_LEAN_AND_ಸರಾಸರಿ
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#ಇಲ್ಲದಿದ್ದರೆ
#   include <dlfcn.h>
#   include <unistd.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಸೂಚನೆ clang diagnostic ignored "-Wgnu-anonymous-struct"
#ಸೂಚನೆ clang diagnostic ignored "-Wmissing-prototypes"
#ಸೂಚನೆ clang diagnostic ignored "-Wsign-compare"
#ಸೂಚನೆ clang diagnostic ignored "-Wlanguage-extension-token"
#ಸೂಚನೆ clang diagnostic ignored "-Wmicrosoft-enum-value"
#ಸೂಚನೆ clang diagnostic ignored "-Wnested-anon-types"

#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "htp-drv.h"
#ಅನ್ತರ್ಭಾವ "libdl.h"

#ಅನ್ತರ್ಭಾವ <domain.h>

//
// Driver API types
//

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ * (*rpcmem_alloc_pfn_t)(ಪೂರ್ಣಾಂಕ heapid, uint32_t flags, ಪೂರ್ಣಾಂಕ size);
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ * (*rpcmem_alloc2_pfn_t)(ಪೂರ್ಣಾಂಕ heapid, uint32_t flags, size_t size);
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ   (*rpcmem_free_pfn_t)(ಶೂನ್ಯ * po);
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಪೂರ್ಣಾಂಕ    (*rpcmem_ಗಮ್ಯ_fd_pfn_t)(ಶೂನ್ಯ * po);

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ AEEResult (*dspqueue_create_pfn_t)(ಪೂರ್ಣಾಂಕ                 domain,
                                           uint32_t            flags,
                                           uint32_t            req_ಸರತಿ_ಗಾತ್ರ,
                                           uint32_t            resp_ಸರತಿ_ಗಾತ್ರ,
                                           dspqueue_callback_t packet_callback,
                                           dspqueue_callback_t error_callback,
                                           ಶೂನ್ಯ *              callback_context,
                                           dspqueue_t *        queue);
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ AEEResult (*dspqueue_close_pfn_t)(dspqueue_t queue);
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ AEEResult (*dspqueue_export_pfn_t)(dspqueue_t queue, uint64_t *ಸರತಿ_id);
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ AEEResult (*dspqueue_write_pfn_t)(dspqueue_t queue, uint32_t flags,
                                          uint32_t ಅಂಕೆ_buffers,
                                          ರಚನೆ dspqueue_ಬಫರ್ *buffers,
                                          uint32_t ಸಂದೇಶ_ಉದ್ದಳತೆ,
                                          ಸ್ಥಿರಾಂಕ uint8_t *ಸಂದೇಶ,
                                          uint32_t timeout_us);
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ AEEResult (*dspqueue_read_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                         uint32_t max_buffers, uint32_t *ಅಂಕೆ_buffers,
                                         ರಚನೆ dspqueue_ಬಫರ್ *buffers,
                                         uint32_t max_ಸಂದೇಶ_ಉದ್ದಳತೆ,
                                         uint32_t *ಸಂದೇಶ_ಉದ್ದಳತೆ, uint8_t *ಸಂದೇಶ,
                                         uint32_t timeout_us);
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ AEEResult (*dspqueue_read_noblock_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                                 uint32_t max_buffers, uint32_t *ಅಂಕೆ_buffers,
                                                 ರಚನೆ dspqueue_ಬಫರ್ *buffers,
                                                 uint32_t max_ಸಂದೇಶ_ಉದ್ದಳತೆ,
                                                 uint32_t *ಸಂದೇಶ_ಉದ್ದಳತೆ, uint8_t *ಸಂದೇಶ);
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಪೂರ್ಣಾಂಕ (*fastrpc_mmap_pfn_t)(ಪೂರ್ಣಾಂಕ domain, ಪೂರ್ಣಾಂಕ fd, ಶೂನ್ಯ *addr, ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ, size_t ಉದ್ದಳತೆ, ಎಣಿಕೆ fastrpc_map_flags flags);
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಪೂರ್ಣಾಂಕ (*fastrpc_munmap_pfn_t)(ಪೂರ್ಣಾಂಕ domain, ಪೂರ್ಣಾಂಕ fd, ಶೂನ್ಯ *addr, size_t ಉದ್ದಳತೆ);

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಪೂರ್ಣಾಂಕ (*remote_handle64_open_pfn_t)(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* name, remote_handle64 *ph);
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಪೂರ್ಣಾಂಕ (*remote_handle64_invoke_pfn_t)(remote_handle64 h, uint32_t dwScalars, remote_arg *pra);
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಪೂರ್ಣಾಂಕ (*remote_handle64_close_pfn_t)(remote_handle h);
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಪೂರ್ಣಾಂಕ (*remote_handle_control_pfn_t)(uint32_t req, ಶೂನ್ಯ* ದತ್ತಾಂಶ, uint32_t datalen);
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಪೂರ್ಣಾಂಕ (*remote_handle64_control_pfn_t)(remote_handle64 h, uint32_t req, ಶೂನ್ಯ* ದತ್ತಾಂಶ, uint32_t datalen);
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಪೂರ್ಣಾಂಕ (*remote_session_control_pfn_t)(uint32_t req, ಶೂನ್ಯ *ದತ್ತಾಂಶ, uint32_t datalen);
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಪೂರ್ಣಾಂಕ (*remote_system_request_pfn_t)(system_req_payload * req);

//
// Driver API pfns
//

rpcmem_alloc_pfn_t  rpcmem_alloc_pfn  = ಶೂನ್ಯಸೂಚಕ;
rpcmem_alloc2_pfn_t rpcmem_alloc2_pfn = ಶೂನ್ಯಸೂಚಕ;
rpcmem_free_pfn_t   rpcmem_free_pfn   = ಶೂನ್ಯಸೂಚಕ;
rpcmem_ಗಮ್ಯ_fd_pfn_t  rpcmem_ಗಮ್ಯ_fd_pfn  = ಶೂನ್ಯಸೂಚಕ;

fastrpc_mmap_pfn_t   fastrpc_mmap_pfn   = ಶೂನ್ಯಸೂಚಕ;
fastrpc_munmap_pfn_t fastrpc_munmap_pfn = ಶೂನ್ಯಸೂಚಕ;

dspqueue_create_pfn_t       dspqueue_create_pfn       = ಶೂನ್ಯಸೂಚಕ;
dspqueue_close_pfn_t        dspqueue_close_pfn        = ಶೂನ್ಯಸೂಚಕ;
dspqueue_export_pfn_t       dspqueue_export_pfn       = ಶೂನ್ಯಸೂಚಕ;
dspqueue_write_pfn_t        dspqueue_write_pfn        = ಶೂನ್ಯಸೂಚಕ;
dspqueue_read_pfn_t         dspqueue_read_pfn         = ಶೂನ್ಯಸೂಚಕ;
dspqueue_read_noblock_pfn_t dspqueue_read_noblock_pfn = ಶೂನ್ಯಸೂಚಕ;

remote_handle64_open_pfn_t    remote_handle64_open_pfn    = ಶೂನ್ಯಸೂಚಕ;
remote_handle64_invoke_pfn_t  remote_handle64_invoke_pfn  = ಶೂನ್ಯಸೂಚಕ;
remote_handle64_close_pfn_t   remote_handle64_close_pfn   = ಶೂನ್ಯಸೂಚಕ;
remote_handle_control_pfn_t   remote_handle_control_pfn   = ಶೂನ್ಯಸೂಚಕ;
remote_handle64_control_pfn_t remote_handle64_control_pfn = ಶೂನ್ಯಸೂಚಕ;
remote_session_control_pfn_t  remote_session_control_pfn  = ಶೂನ್ಯಸೂಚಕ;
remote_system_request_pfn_t   remote_system_request_pfn   = ಶೂನ್ಯಸೂಚಕ;

//
// Driver API
//

ಶೂನ್ಯ * rpcmem_alloc(ಪೂರ್ಣಾಂಕ heapid, uint32_t flags, ಪೂರ್ಣಾಂಕ size) {
    ಹಿಂತಿರುಗಿ rpcmem_alloc_pfn(heapid, flags, size);
}

ಶೂನ್ಯ * rpcmem_alloc2(ಪೂರ್ಣಾಂಕ heapid, uint32_t flags, size_t size) {
    ಆದರೆ (rpcmem_alloc2_pfn) {
        ಹಿಂತಿರುಗಿ rpcmem_alloc2_pfn(heapid, flags, size);
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_LOG_INFO("ggml-hex: rpcmem_alloc2 not found, falling back to rpcmem_alloc\n");
        ಹಿಂತಿರುಗಿ rpcmem_alloc_pfn(heapid, flags, size);
    }
}

ಶೂನ್ಯ rpcmem_free(ಶೂನ್ಯ * po) {
    ಹಿಂತಿರುಗಿ rpcmem_free_pfn(po);
}

ಪೂರ್ಣಾಂಕ rpcmem_ಗಮ್ಯ_fd(ಶೂನ್ಯ * po) {
    ಹಿಂತಿರುಗಿ rpcmem_ಗಮ್ಯ_fd_pfn(po);
}

HTPDRV_API ಪೂರ್ಣಾಂಕ fastrpc_mmap(ಪೂರ್ಣಾಂಕ domain, ಪೂರ್ಣಾಂಕ fd, ಶೂನ್ಯ * addr, ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ, size_t ಉದ್ದಳತೆ, ಎಣಿಕೆ fastrpc_map_flags flags) {
    ಹಿಂತಿರುಗಿ fastrpc_mmap_pfn(domain, fd, addr, ಸ್ಥಾನಾಂತರ, ಉದ್ದಳತೆ, flags);
}

HTPDRV_API ಪೂರ್ಣಾಂಕ fastrpc_munmap(ಪೂರ್ಣಾಂಕ domain, ಪೂರ್ಣಾಂಕ fd, ಶೂನ್ಯ * addr, size_t ಉದ್ದಳತೆ) {
    ಹಿಂತಿರುಗಿ fastrpc_munmap_pfn(domain, fd, addr, ಉದ್ದಳತೆ);
}

AEEResult dspqueue_create(ಪೂರ್ಣಾಂಕ                 domain,
                          uint32_t            flags,
                          uint32_t            req_ಸರತಿ_ಗಾತ್ರ,
                          uint32_t            resp_ಸರತಿ_ಗಾತ್ರ,
                          dspqueue_callback_t packet_callback,
                          dspqueue_callback_t error_callback,
                          ಶೂನ್ಯ *              callback_context,
                          dspqueue_t *        queue) {
    ಹಿಂತಿರುಗಿ dspqueue_create_pfn(domain, flags, req_ಸರತಿ_ಗಾತ್ರ, resp_ಸರತಿ_ಗಾತ್ರ, packet_callback, error_callback,
                               callback_context, queue);
}

AEEResult dspqueue_close(dspqueue_t queue) {
    ಹಿಂತಿರುಗಿ dspqueue_close_pfn(queue);
}

AEEResult dspqueue_export(dspqueue_t queue, uint64_t * ಸರತಿ_id) {
    ಹಿಂತಿರುಗಿ dspqueue_export_pfn(queue, ಸರತಿ_id);
}

AEEResult dspqueue_write(dspqueue_t               queue,
                         uint32_t                 flags,
                         uint32_t                 ಅಂಕೆ_buffers,
                         ರಚನೆ dspqueue_ಬಫರ್ * buffers,
                         uint32_t                 ಸಂದೇಶ_ಉದ್ದಳತೆ,
                         ಸ್ಥಿರಾಂಕ uint8_t *          ಸಂದೇಶ,
                         uint32_t                 timeout_us) {
    ಹಿಂತಿರುಗಿ dspqueue_write_pfn(queue, flags, ಅಂಕೆ_buffers, buffers, ಸಂದೇಶ_ಉದ್ದಳತೆ, ಸಂದೇಶ, timeout_us);
}

AEEResult dspqueue_read(dspqueue_t               queue,
                        uint32_t *               flags,
                        uint32_t                 max_buffers,
                        uint32_t *               ಅಂಕೆ_buffers,
                        ರಚನೆ dspqueue_ಬಫರ್ * buffers,
                        uint32_t                 max_ಸಂದೇಶ_ಉದ್ದಳತೆ,
                        uint32_t *               ಸಂದೇಶ_ಉದ್ದಳತೆ,
                        uint8_t *                ಸಂದೇಶ,
                        uint32_t                 timeout_us) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
    ಆದರೆ (timeout_us == 0) {
        ಹಿಂತಿರುಗಿ dspqueue_read_noblock_pfn(queue, flags, max_buffers, ಅಂಕೆ_buffers, buffers, max_ಸಂದೇಶ_ಉದ್ದಳತೆ,
                                     ಸಂದೇಶ_ಉದ್ದಳತೆ, ಸಂದೇಶ);
    }
#ಆದರೆ_ಅಂತ್ಯ
    ಹಿಂತಿರುಗಿ dspqueue_read_pfn(queue, flags, max_buffers, ಅಂಕೆ_buffers, buffers, max_ಸಂದೇಶ_ಉದ್ದಳತೆ, ಸಂದೇಶ_ಉದ್ದಳತೆ,
                             ಸಂದೇಶ, timeout_us);
}

HTPDRV_API ಪೂರ್ಣಾಂಕ remote_handle64_open(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name, remote_handle64 * ph) {
    ಹಿಂತಿರುಗಿ remote_handle64_open_pfn(name, ph);
}

HTPDRV_API ಪೂರ್ಣಾಂಕ remote_handle64_invoke(remote_handle64 h, uint32_t dwScalars, remote_arg * pra) {
    ಹಿಂತಿರುಗಿ remote_handle64_invoke_pfn(h, dwScalars, pra);
}

HTPDRV_API ಪೂರ್ಣಾಂಕ remote_handle64_close(remote_handle64 h) {
    ಹಿಂತಿರುಗಿ remote_handle64_close_pfn(h);
}

HTPDRV_API ಪೂರ್ಣಾಂಕ remote_handle_control(uint32_t req, ಶೂನ್ಯ * ದತ್ತಾಂಶ, uint32_t datalen) {
    ಹಿಂತಿರುಗಿ remote_handle_control_pfn(req, ದತ್ತಾಂಶ, datalen);
}

HTPDRV_API ಪೂರ್ಣಾಂಕ remote_handle64_control(remote_handle64 h, uint32_t req, ಶೂನ್ಯ * ದತ್ತಾಂಶ, uint32_t datalen) {
    ಹಿಂತಿರುಗಿ remote_handle64_control_pfn(h, req, ದತ್ತಾಂಶ, datalen);
}

HTPDRV_API ಪೂರ್ಣಾಂಕ remote_session_control(uint32_t req, ಶೂನ್ಯ * ದತ್ತಾಂಶ, uint32_t datalen) {
    ಹಿಂತಿರುಗಿ remote_session_control_pfn(req, ದತ್ತಾಂಶ, datalen);
}

HTPDRV_API ಪೂರ್ಣಾಂಕ remote_system_request(system_req_payload * req) {
    ಆದರೆ (!remote_system_request_pfn) {
        ಹಿಂತಿರುಗಿ AEE_EUNSUPPORTEDAPI;
    }
    ಹಿಂತಿರುಗಿ remote_system_request_pfn(req);
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32

ಸ್ಥಿರ std::string wstr_ಗಮ್ಯ_str(std::wstring_view wstr) {
    std::string ಫಲಿತಾಂಶ;
    ಆದರೆ (wstr.ಖಾಲಿಗಳು()) {
        ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
    }
    ಸ್ವಯಂಚಾಲಿತ ಬೈಟುಗಳು_needed = Wide_Char_ಗಮ್ಯ_Multi_Byte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.ದತ್ತಾಂಶ(), (ಪೂರ್ಣಾಂಕ) wstr.size(),
                                            ಶೂನ್ಯಸೂಚಕ, 0, ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ);
    ಆದರೆ (ಬೈಟುಗಳು_needed == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", Get_ಕೊನೆಯ_Error());
        ಎಸೆ std::runtime_error("Invalid wstring input");
    }

    ಫಲಿತಾಂಶ.resize(ಬೈಟುಗಳು_needed, '\0');
    ಪೂರ್ಣಾಂಕ ಬೈಟುಗಳು_written = Wide_Char_ಗಮ್ಯ_Multi_Byte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.ದತ್ತಾಂಶ(), (ಪೂರ್ಣಾಂಕ) wstr.size(),
                                            ಫಲಿತಾಂಶ.ದತ್ತಾಂಶ(), ಬೈಟುಗಳು_needed,
                                            ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ);
    ಆದರೆ (ಬೈಟುಗಳು_written == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", Get_ಕೊನೆಯ_Error());
        ಎಸೆ std::runtime_error("Wstring conversion failed");
    }
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ಸ್ಥಿರ std::string get_driver_ಮಾರ್ಗ() {
    std::wstring serviceName = L"qcnspmcdm";
    std::string ಫಲಿತಾಂಶ;

    // Get a handle to the SCM database.
    SC_HANDLE schSCManager = OpenSCManagerW(ಖಾಲಿ, ಖಾಲಿ, STANDARD_RIGHTS_READ);
    ಆದರೆ (ಶೂನ್ಯಸೂಚಕ == schSCManager) {
        GGML_LOG_ERROR("ggml-hex: Failed to open SCManager. Error: %lu\n", Get_ಕೊನೆಯ_Error());
        ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
    }

    // Get a handle to the service.
    SC_HANDLE schService = OpenServiceW(schSCManager,           // SCM database
                                        serviceName.c_str(),    // name of service
                                        SERVICE_QUERY_CONFIG);  // need query config access

    ಆದರೆ (ಶೂನ್ಯಸೂಚಕ == schService) {
        GGML_LOG_ERROR("ggml-hex: Failed to open qcnspmcdm service. Error: %lu\n", Get_ಕೊನೆಯ_Error());
        CloseServiceHandle(schSCManager);
        ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
    }

    // Store the size of buffer used as an output.
    DWORD ಬಫರ್_ಗಾತ್ರ_2;
    ಆದರೆ (!QueryServiceConfigW(schService, ಖಾಲಿ, 0, &ಬಫರ್_ಗಾತ್ರ_2) &&
        (Get_ಕೊನೆಯ_Error() != ERROR_INSUFFICIENT_ಬಫರ್)) {
        GGML_LOG_ERROR("ggml-hex: Failed to query service config. Error: %lu\n", Get_ಕೊನೆಯ_Error());
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
    }
    // Get the configuration of the service.
    LPQUERY_SERVICE_CONFIGW serviceConfig =
        static_cast<LPQUERY_SERVICE_CONFIGW>(LocalAlloc(LMEM_FIXED, ಬಫರ್_ಗಾತ್ರ_2));
    ಆದರೆ (!QueryServiceConfigW(schService, serviceConfig, ಬಫರ್_ಗಾತ್ರ_2, &ಬಫರ್_ಗಾತ್ರ_2)) {
        fprintf(stderr, "ggml-hex: Failed to query service config. Error: %lu\n", Get_ಕೊನೆಯ_Error());
        LocalFree(serviceConfig);
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
    }

    // Read the driver file path get its parent directory
    std::wstring driver_ಮಾರ್ಗ = std::wstring(serviceConfig->lp_ದ್ವಿಮಾನ_ಮಾರ್ಗ_Name);
    driver_ಮಾರ್ಗ = driver_ಮಾರ್ಗ.substr(0, driver_ಮಾರ್ಗ.find_last_of(L"\\"));

    // Clean up resources
    LocalFree(serviceConfig);
    CloseServiceHandle(schService);
    CloseServiceHandle(schSCManager);

    // Driver path would contain invalid path string, like:
    // \SystemRoot\System32\DriverStore\FileRepository\qcadsprpc8280.inf_arm64_c2b9460c9a072f37
    // "\SystemRoot" should be replace with a correct one (e.g. C:\Windows)
    ಸ್ಥಿರಾಂಕ std::wstring system_ಬೇರು_Placeholder = L"\\SystemRoot";
    ಆದರೆ (0 != driver_ಮಾರ್ಗ.compare(0, system_ಬೇರು_Placeholder.ಉದ್ದಳತೆ(), system_ಬೇರು_Placeholder)) {
        GGML_LOG_ERROR("ggml-hex: String pattern not found in driver path.\n");
        ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
    }

    // Replace \SystemRoot with an absolute path from system ENV windir
    ಸ್ಥಿರಾಂಕ std::wstring system_ಬೇರು_Env = L"windir";

    // Query the number of wide characters this variable requires
    DWORD ಅಂಕೆ_ಪದಗಳು = GetEnvironmentVariableW(system_ಬೇರು_Env.c_str(), ಖಾಲಿ, 0);
    ಆದರೆ (ಅಂಕೆ_ಪದಗಳು == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed get systemRoot environment variable\n");
        ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
    }

    // Query the actual system root name from environment variable
    std::vector<wchar_t> system_ಬೇರು(ಅಂಕೆ_ಪದಗಳು + 1);
    ಅಂಕೆ_ಪದಗಳು = GetEnvironmentVariableW(system_ಬೇರು_Env.c_str(), system_ಬೇರು.ದತ್ತಾಂಶ(), ಅಂಕೆ_ಪದಗಳು + 1);
    ಆದರೆ (ಅಂಕೆ_ಪದಗಳು == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed to read windir environment variable\n");
        ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
    }
    driver_ಮಾರ್ಗ.replace(0, system_ಬೇರು_Placeholder.ಉದ್ದಳತೆ(), std::wstring(system_ಬೇರು.ದತ್ತಾಂಶ()));

    ಹಿಂತಿರುಗಿ wstr_ಗಮ್ಯ_str(driver_ಮಾರ್ಗ);
}

#ಆದರೆ_ಅಂತ್ಯ

ಬಳಸಿ dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

ಪೂರ್ಣಾಂಕ htpdrv_init() {
    ಸ್ಥಿರ dl_handle_ptr lib_cdsp_rpc_handle = ಶೂನ್ಯಸೂಚಕ;
    ಸ್ಥಿರ ತಾರ್ಕಿಕ initialized = ಸುಳ್ಳು;
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
    std::string drv_ಮಾರ್ಗ = get_driver_ಮಾರ್ಗ() + "\\" + "libcdsprpc.dll";
#ಇಲ್ಲದಿದ್ದರೆ
    std::string drv_ಮಾರ್ಗ = "libcdsprpc.so";
#ಆದರೆ_ಅಂತ್ಯ
    ಆದರೆ (initialized) {
        GGML_LOG_INFO("ggml-hex: Driver already loaded\n");
        ಹಿಂತಿರುಗಿ AEE_SUCCESS;
    }
    GGML_LOG_INFO("ggml-hex: Loading driver %s\n", drv_ಮಾರ್ಗ.c_str());

    fs::path path{ drv_ಮಾರ್ಗ.c_str() };
    dl_handle_ptr handle { dl_ತುಂಬು_library(path) };
    ಆದರೆ (!handle) {
        GGML_LOG_ERROR("ggml-hex: failed to load %s: %s\n", path.u8string().c_str(), dl_error());
        ಹಿಂತಿರುಗಿ AEE_EUNABLETOLOAD;
    }

#ವ್ಯಾಖ್ಯಾನಿಸು dlsym(drv, type, pfn, ಚಿಹ್ನೆ, ignore)                               \
    ಮಾಡು {                                                                    \
        pfn = (type) dl_get_sym(drv, #symbol);                              \
        ಆದರೆ (!ignore && ಶೂನ್ಯಸೂಚಕ == pfn) {                                    \
            GGML_LOG_ERROR("ggml-hex: failed to dlsym %s\n", #symbol);      \
            ಹಿಂತಿರುಗಿ AEE_EUNABLETOLOAD;                                       \
        }                                                                   \
    } ಆಗ (0)

    dlsym(handle.get(), rpcmem_alloc_pfn_t, rpcmem_alloc_pfn, rpcmem_alloc, ಸುಳ್ಳು);
    dlsym(handle.get(), rpcmem_alloc2_pfn_t, rpcmem_alloc2_pfn, rpcmem_alloc2, ನಿಜ);
    dlsym(handle.get(), rpcmem_free_pfn_t, rpcmem_free_pfn, rpcmem_free, ಸುಳ್ಳು);
    dlsym(handle.get(), rpcmem_ಗಮ್ಯ_fd_pfn_t, rpcmem_ಗಮ್ಯ_fd_pfn, rpcmem_ಗಮ್ಯ_fd, ಸುಳ್ಳು);
    dlsym(handle.get(), fastrpc_mmap_pfn_t, fastrpc_mmap_pfn, fastrpc_mmap, ಸುಳ್ಳು);
    dlsym(handle.get(), fastrpc_munmap_pfn_t, fastrpc_munmap_pfn, fastrpc_munmap, ಸುಳ್ಳು);
    dlsym(handle.get(), dspqueue_create_pfn_t, dspqueue_create_pfn, dspqueue_create, ಸುಳ್ಳು);
    dlsym(handle.get(), dspqueue_close_pfn_t, dspqueue_close_pfn, dspqueue_close, ಸುಳ್ಳು);
    dlsym(handle.get(), dspqueue_export_pfn_t, dspqueue_export_pfn, dspqueue_export, ಸುಳ್ಳು);
    dlsym(handle.get(), dspqueue_write_pfn_t, dspqueue_write_pfn, dspqueue_write, ಸುಳ್ಳು);
    dlsym(handle.get(), dspqueue_read_pfn_t, dspqueue_read_pfn, dspqueue_read, ಸುಳ್ಳು);
    dlsym(handle.get(), dspqueue_read_noblock_pfn_t, dspqueue_read_noblock_pfn, dspqueue_read_noblock, ಸುಳ್ಳು);
    dlsym(handle.get(), remote_handle64_open_pfn_t, remote_handle64_open_pfn, remote_handle64_open, ಸುಳ್ಳು);
    dlsym(handle.get(), remote_handle64_invoke_pfn_t, remote_handle64_invoke_pfn, remote_handle64_invoke, ಸುಳ್ಳು);
    dlsym(handle.get(), remote_handle_control_pfn_t, remote_handle_control_pfn, remote_handle_control, ಸುಳ್ಳು);
    dlsym(handle.get(), remote_handle64_control_pfn_t, remote_handle64_control_pfn, remote_handle64_control, ಸುಳ್ಳು);
    dlsym(handle.get(), remote_session_control_pfn_t, remote_session_control_pfn, remote_session_control, ಸುಳ್ಳು);
    dlsym(handle.get(), remote_handle64_close_pfn_t, remote_handle64_close_pfn, remote_handle64_close, ಸುಳ್ಳು);
    dlsym(handle.get(), remote_system_request_pfn_t, remote_system_request_pfn, remote_system_request, ನಿಜ);

    lib_cdsp_rpc_handle = std::move(handle);
    initialized         = ನಿಜ;

    ಹಿಂತಿರುಗಿ AEE_SUCCESS;
}

domain * htpdrv_get_domain(ಪೂರ್ಣಾಂಕ domain_id) {
    ಪೂರ್ಣಾಂಕ i    = 0;
    ಪೂರ್ಣಾಂಕ size = ಗಾತ್ರ(supported_domains) / ಗಾತ್ರ(domain);

    ಗಾಗಿ (i = 0; i < size; i++) {
        ಆದರೆ (supported_domains[i].id == domain_id) {
            ಹಿಂತಿರುಗಿ &supported_domains[i];
        }
    }

    ಹಿಂತಿರುಗಿ ಖಾಲಿ;
}

ಪೂರ್ಣಾಂಕ htpdrv_get_arch(ಪೂರ್ಣಾಂಕ domain, ಪೂರ್ಣಾಂಕ * arch) {
    ಆದರೆ (!remote_handle_control_pfn) {
        GGML_LOG_ERROR("ggml-hex: remote_handle_control is not supported on this device\n");
        ಹಿಂತಿರುಗಿ AEE_EUNSUPPORTEDAPI;
    }

    ರಚನೆ remote_dsp_capability arch_ver;
    arch_ver.domain       = (uint32_t) domain;
    arch_ver.attribute_ID = ARCH_VER;
    arch_ver.capability   = (uint32_t) 0;

    ಪೂರ್ಣಾಂಕ err = remote_handle_control(DSPRPC_GET_DSP_INFO, &arch_ver, ಗಾತ್ರ(arch_ver));
    ಆದರೆ ((err & 0xff) == (AEE_EUNSUPPORTEDAPI & 0xff)) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability API is not supported on this device\n");
        ಹಿಂತಿರುಗಿ AEE_EUNSUPPORTEDAPI;
    }

    ಆದರೆ (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability query failed (err %d)\n", err);
        ಹಿಂತಿರುಗಿ err;
    }

    uint32_t val = arch_ver.capability & 0xff;
    *arch = (ಪೂರ್ಣಾಂಕ) ((val >> 4) * 10 + (val & 0x0f));
    ಹಿಂತಿರುಗಿ 0;
}
