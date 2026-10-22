#அந்தர்பாவ <filesystem>
#அந்தர்பாவ <set>
#அந்தர்பாவ <sstream>
#அந்தர்பாவ <string>

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#இல்லையெனில்
#   include <dlfcn.h>
#   include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"
#pragma clang diagnostic ignored "-Wnested-anon-types"

#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "htp-drv.h"
#அந்தர்பாவ "libdl.h"

#அந்தர்பாவ <domain.h>

//
// Driver API types
//

வகைவரை வெறுமை * (*rpcmem_alloc_pfn_t)(முழு heapid, uint32_t flags, முழு size);
வகைவரை வெறுமை * (*rpcmem_alloc2_pfn_t)(முழு heapid, uint32_t flags, size_t size);
வகைவரை வெறுமை   (*rpcmem_free_pfn_t)(வெறுமை * po);
வகைவரை முழு    (*rpcmem_to_fd_pfn_t)(வெறுமை * po);

வகைவரை AEEResult (*dspqueue_create_pfn_t)(முழு                 domain,
                                           uint32_t            flags,
                                           uint32_t            req_queue_size,
                                           uint32_t            resp_queue_size,
                                           dspqueue_callback_t packet_callback,
                                           dspqueue_callback_t error_callback,
                                           வெறுமை *              callback_context,
                                           dspqueue_t *        queue);
வகைவரை AEEResult (*dspqueue_close_pfn_t)(dspqueue_t queue);
வகைவரை AEEResult (*dspqueue_export_pfn_t)(dspqueue_t queue, uint64_t *queue_id);
வகைவரை AEEResult (*dspqueue_write_pfn_t)(dspqueue_t queue, uint32_t flags,
                                          uint32_t இலக்கம்_buffers,
                                          கட்டமைப்பு dspqueue_buffer *buffers,
                                          uint32_t message_length,
                                          மாறிலி uint8_t *message,
                                          uint32_t timeout_us);
வகைவரை AEEResult (*dspqueue_read_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                         uint32_t max_buffers, uint32_t *இலக்கம்_buffers,
                                         கட்டமைப்பு dspqueue_buffer *buffers,
                                         uint32_t max_message_length,
                                         uint32_t *message_length, uint8_t *message,
                                         uint32_t timeout_us);
வகைவரை AEEResult (*dspqueue_read_noblock_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                                 uint32_t max_buffers, uint32_t *இலக்கம்_buffers,
                                                 கட்டமைப்பு dspqueue_buffer *buffers,
                                                 uint32_t max_message_length,
                                                 uint32_t *message_length, uint8_t *message);
வகைவரை முழு (*fastrpc_mmap_pfn_t)(முழு domain, முழு fd, வெறுமை *addr, முழு offset, size_t length, எண்ணிக்கை fastrpc_map_flags flags);
வகைவரை முழு (*fastrpc_munmap_pfn_t)(முழு domain, முழு fd, வெறுமை *addr, size_t length);

வகைவரை முழு (*remote_handle64_open_pfn_t)(மாறிலி எழுத்து* name, remote_handle64 *ph);
வகைவரை முழு (*remote_handle64_invoke_pfn_t)(remote_handle64 h, uint32_t dwScalars, remote_arg *pra);
வகைவரை முழு (*remote_handle64_close_pfn_t)(remote_handle h);
வகைவரை முழு (*remote_handle_control_pfn_t)(uint32_t req, வெறுமை* data, uint32_t datalen);
வகைவரை முழு (*remote_handle64_control_pfn_t)(remote_handle64 h, uint32_t req, வெறுமை* data, uint32_t datalen);
வகைவரை முழு (*remote_session_control_pfn_t)(uint32_t req, வெறுமை *data, uint32_t datalen);
வகைவரை முழு (*remote_system_request_pfn_t)(system_req_payload * req);

//
// Driver API pfns
//

rpcmem_alloc_pfn_t  rpcmem_alloc_pfn  = nullptr;
rpcmem_alloc2_pfn_t rpcmem_alloc2_pfn = nullptr;
rpcmem_free_pfn_t   rpcmem_free_pfn   = nullptr;
rpcmem_to_fd_pfn_t  rpcmem_to_fd_pfn  = nullptr;

fastrpc_mmap_pfn_t   fastrpc_mmap_pfn   = nullptr;
fastrpc_munmap_pfn_t fastrpc_munmap_pfn = nullptr;

dspqueue_create_pfn_t       dspqueue_create_pfn       = nullptr;
dspqueue_close_pfn_t        dspqueue_close_pfn        = nullptr;
dspqueue_export_pfn_t       dspqueue_export_pfn       = nullptr;
dspqueue_write_pfn_t        dspqueue_write_pfn        = nullptr;
dspqueue_read_pfn_t         dspqueue_read_pfn         = nullptr;
dspqueue_read_noblock_pfn_t dspqueue_read_noblock_pfn = nullptr;

remote_handle64_open_pfn_t    remote_handle64_open_pfn    = nullptr;
remote_handle64_invoke_pfn_t  remote_handle64_invoke_pfn  = nullptr;
remote_handle64_close_pfn_t   remote_handle64_close_pfn   = nullptr;
remote_handle_control_pfn_t   remote_handle_control_pfn   = nullptr;
remote_handle64_control_pfn_t remote_handle64_control_pfn = nullptr;
remote_session_control_pfn_t  remote_session_control_pfn  = nullptr;
remote_system_request_pfn_t   remote_system_request_pfn   = nullptr;

//
// Driver API
//

வெறுமை * rpcmem_alloc(முழு heapid, uint32_t flags, முழு size) {
    திரும்பு rpcmem_alloc_pfn(heapid, flags, size);
}

வெறுமை * rpcmem_alloc2(முழு heapid, uint32_t flags, size_t size) {
    எனில் (rpcmem_alloc2_pfn) {
        திரும்பு rpcmem_alloc2_pfn(heapid, flags, size);
    } இல்லையெனில் {
        GGML_LOG_INFO("ggml-hex: rpcmem_alloc2 not found, falling back to rpcmem_alloc\n");
        திரும்பு rpcmem_alloc_pfn(heapid, flags, size);
    }
}

வெறுமை rpcmem_free(வெறுமை * po) {
    திரும்பு rpcmem_free_pfn(po);
}

முழு rpcmem_to_fd(வெறுமை * po) {
    திரும்பு rpcmem_to_fd_pfn(po);
}

HTPDRV_API முழு fastrpc_mmap(முழு domain, முழு fd, வெறுமை * addr, முழு offset, size_t length, எண்ணிக்கை fastrpc_map_flags flags) {
    திரும்பு fastrpc_mmap_pfn(domain, fd, addr, offset, length, flags);
}

HTPDRV_API முழு fastrpc_munmap(முழு domain, முழு fd, வெறுமை * addr, size_t length) {
    திரும்பு fastrpc_munmap_pfn(domain, fd, addr, length);
}

AEEResult dspqueue_create(முழு                 domain,
                          uint32_t            flags,
                          uint32_t            req_queue_size,
                          uint32_t            resp_queue_size,
                          dspqueue_callback_t packet_callback,
                          dspqueue_callback_t error_callback,
                          வெறுமை *              callback_context,
                          dspqueue_t *        queue) {
    திரும்பு dspqueue_create_pfn(domain, flags, req_queue_size, resp_queue_size, packet_callback, error_callback,
                               callback_context, queue);
}

AEEResult dspqueue_close(dspqueue_t queue) {
    திரும்பு dspqueue_close_pfn(queue);
}

AEEResult dspqueue_export(dspqueue_t queue, uint64_t * queue_id) {
    திரும்பு dspqueue_export_pfn(queue, queue_id);
}

AEEResult dspqueue_write(dspqueue_t               queue,
                         uint32_t                 flags,
                         uint32_t                 இலக்கம்_buffers,
                         கட்டமைப்பு dspqueue_buffer * buffers,
                         uint32_t                 message_length,
                         மாறிலி uint8_t *          message,
                         uint32_t                 timeout_us) {
    திரும்பு dspqueue_write_pfn(queue, flags, இலக்கம்_buffers, buffers, message_length, message, timeout_us);
}

AEEResult dspqueue_read(dspqueue_t               queue,
                        uint32_t *               flags,
                        uint32_t                 max_buffers,
                        uint32_t *               இலக்கம்_buffers,
                        கட்டமைப்பு dspqueue_buffer * buffers,
                        uint32_t                 max_message_length,
                        uint32_t *               message_length,
                        uint8_t *                message,
                        uint32_t                 timeout_us) {
#ifdef _WIN32
    எனில் (timeout_us == 0) {
        திரும்பு dspqueue_read_noblock_pfn(queue, flags, max_buffers, இலக்கம்_buffers, buffers, max_message_length,
                                     message_length, message);
    }
#endif
    திரும்பு dspqueue_read_pfn(queue, flags, max_buffers, இலக்கம்_buffers, buffers, max_message_length, message_length,
                             message, timeout_us);
}

HTPDRV_API முழு remote_handle64_open(மாறிலி எழுத்து * name, remote_handle64 * ph) {
    திரும்பு remote_handle64_open_pfn(name, ph);
}

HTPDRV_API முழு remote_handle64_invoke(remote_handle64 h, uint32_t dwScalars, remote_arg * pra) {
    திரும்பு remote_handle64_invoke_pfn(h, dwScalars, pra);
}

HTPDRV_API முழு remote_handle64_close(remote_handle64 h) {
    திரும்பு remote_handle64_close_pfn(h);
}

HTPDRV_API முழு remote_handle_control(uint32_t req, வெறுமை * data, uint32_t datalen) {
    திரும்பு remote_handle_control_pfn(req, data, datalen);
}

HTPDRV_API முழு remote_handle64_control(remote_handle64 h, uint32_t req, வெறுமை * data, uint32_t datalen) {
    திரும்பு remote_handle64_control_pfn(h, req, data, datalen);
}

HTPDRV_API முழு remote_session_control(uint32_t req, வெறுமை * data, uint32_t datalen) {
    திரும்பு remote_session_control_pfn(req, data, datalen);
}

HTPDRV_API முழு remote_system_request(system_req_payload * req) {
    எனில் (!remote_system_request_pfn) {
        திரும்பு AEE_EUNSUPPORTEDAPI;
    }
    திரும்பு remote_system_request_pfn(req);
}

#ifdef _WIN32

நிலையான std::string wstr_to_str(std::wstring_view wstr) {
    std::string result;
    எனில் (wstr.empty()) {
        திரும்பு result;
    }
    auto bytes_needed = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.data(), (முழு) wstr.size(),
                                            nullptr, 0, nullptr, nullptr);
    எனில் (bytes_needed == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        எறி std::runtime_error("Invalid wstring input");
    }

    result.resize(bytes_needed, '\0');
    முழு bytes_written = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.data(), (முழு) wstr.size(),
                                            result.data(), bytes_needed,
                                            nullptr, nullptr);
    எனில் (bytes_written == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        எறி std::runtime_error("Wstring conversion failed");
    }
    திரும்பு result;
}

நிலையான std::string get_driver_path() {
    std::wstring serviceName = L"qcnspmcdm";
    std::string result;

    // Get a handle to the SCM database.
    SC_HANDLE schSCManager = OpenSCManagerW(வெற்று, வெற்று, STANDARD_RIGHTS_READ);
    எனில் (nullptr == schSCManager) {
        GGML_LOG_ERROR("ggml-hex: Failed to open SCManager. Error: %lu\n", GetLastError());
        திரும்பு result;
    }

    // Get a handle to the service.
    SC_HANDLE schService = OpenServiceW(schSCManager,           // SCM database
                                        serviceName.c_str(),    // name of service
                                        SERVICE_QUERY_CONFIG);  // need query config access

    எனில் (nullptr == schService) {
        GGML_LOG_ERROR("ggml-hex: Failed to open qcnspmcdm service. Error: %lu\n", GetLastError());
        CloseServiceHandle(schSCManager);
        திரும்பு result;
    }

    // Store the size of buffer used as an output.
    DWORD bufferSize;
    எனில் (!QueryServiceConfigW(schService, வெற்று, 0, &bufferSize) &&
        (GetLastError() != ERROR_INSUFFICIENT_BUFFER)) {
        GGML_LOG_ERROR("ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        திரும்பு result;
    }
    // Get the configuration of the service.
    LPQUERY_SERVICE_CONFIGW serviceConfig =
        static_cast<LPQUERY_SERVICE_CONFIGW>(LocalAlloc(LMEM_FIXED, bufferSize));
    எனில் (!QueryServiceConfigW(schService, serviceConfig, bufferSize, &bufferSize)) {
        fprintf(stderr, "ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        LocalFree(serviceConfig);
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        திரும்பு result;
    }

    // Read the driver file path get its parent directory
    std::wstring driverPath = std::wstring(serviceConfig->lp_இருமம்_Path_Name);
    driverPath = driverPath.substr(0, driverPath.find_last_of(L"\\"));

    // Clean up resources
    LocalFree(serviceConfig);
    CloseServiceHandle(schService);
    CloseServiceHandle(schSCManager);

    // Driver path would contain invalid path string, like:
    // \SystemRoot\System32\DriverStore\FileRepository\qcadsprpc8280.inf_arm64_c2b9460c9a072f37
    // "\SystemRoot" should be replace with a correct one (e.g. C:\Windows)
    மாறிலி std::wstring system_வேர்_Placeholder = L"\\SystemRoot";
    எனில் (0 != driverPath.compare(0, system_வேர்_Placeholder.length(), system_வேர்_Placeholder)) {
        GGML_LOG_ERROR("ggml-hex: String pattern not found in driver path.\n");
        திரும்பு result;
    }

    // Replace \SystemRoot with an absolute path from system ENV windir
    மாறிலி std::wstring system_வேர்_Env = L"windir";

    // Query the number of wide characters this variable requires
    DWORD இலக்கம்_சொற்கள் = GetEnvironmentVariableW(system_வேர்_Env.c_str(), வெற்று, 0);
    எனில் (இலக்கம்_சொற்கள் == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed get systemRoot environment variable\n");
        திரும்பு result;
    }

    // Query the actual system root name from environment variable
    std::vector<wchar_t> system_வேர்(இலக்கம்_சொற்கள் + 1);
    இலக்கம்_சொற்கள் = GetEnvironmentVariableW(system_வேர்_Env.c_str(), system_வேர்.data(), இலக்கம்_சொற்கள் + 1);
    எனில் (இலக்கம்_சொற்கள் == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed to read windir environment variable\n");
        திரும்பு result;
    }
    driverPath.replace(0, system_வேர்_Placeholder.length(), std::wstring(system_வேர்.data()));

    திரும்பு wstr_to_str(driverPath);
}

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

முழு htpdrv_init() {
    நிலையான dl_handle_ptr lib_cdsp_rpc_handle = nullptr;
    நிலையான தருக்க initialized = பொய்;
#ifdef _WIN32
    std::string drv_path = get_driver_path() + "\\" + "libcdsprpc.dll";
#இல்லையெனில்
    std::string drv_path = "libcdsprpc.so";
#endif
    எனில் (initialized) {
        GGML_LOG_INFO("ggml-hex: Driver already loaded\n");
        திரும்பு AEE_SUCCESS;
    }
    GGML_LOG_INFO("ggml-hex: Loading driver %s\n", drv_path.c_str());

    fs::path path{ drv_path.c_str() };
    dl_handle_ptr handle { dl_load_library(path) };
    எனில் (!handle) {
        GGML_LOG_ERROR("ggml-hex: failed to load %s: %s\n", path.u8string().c_str(), dl_error());
        திரும்பு AEE_EUNABLETOLOAD;
    }

#define dlsym(drv, type, pfn, குறியீடு, ignore)                               \
    செய் {                                                                    \
        pfn = (type) dl_get_sym(drv, #symbol);                              \
        எனில் (!ignore && nullptr == pfn) {                                    \
            GGML_LOG_ERROR("ggml-hex: failed to dlsym %s\n", #symbol);      \
            திரும்பு AEE_EUNABLETOLOAD;                                       \
        }                                                                   \
    } போது (0)

    dlsym(handle.get(), rpcmem_alloc_pfn_t, rpcmem_alloc_pfn, rpcmem_alloc, பொய்);
    dlsym(handle.get(), rpcmem_alloc2_pfn_t, rpcmem_alloc2_pfn, rpcmem_alloc2, உண்மை);
    dlsym(handle.get(), rpcmem_free_pfn_t, rpcmem_free_pfn, rpcmem_free, பொய்);
    dlsym(handle.get(), rpcmem_to_fd_pfn_t, rpcmem_to_fd_pfn, rpcmem_to_fd, பொய்);
    dlsym(handle.get(), fastrpc_mmap_pfn_t, fastrpc_mmap_pfn, fastrpc_mmap, பொய்);
    dlsym(handle.get(), fastrpc_munmap_pfn_t, fastrpc_munmap_pfn, fastrpc_munmap, பொய்);
    dlsym(handle.get(), dspqueue_create_pfn_t, dspqueue_create_pfn, dspqueue_create, பொய்);
    dlsym(handle.get(), dspqueue_close_pfn_t, dspqueue_close_pfn, dspqueue_close, பொய்);
    dlsym(handle.get(), dspqueue_export_pfn_t, dspqueue_export_pfn, dspqueue_export, பொய்);
    dlsym(handle.get(), dspqueue_write_pfn_t, dspqueue_write_pfn, dspqueue_write, பொய்);
    dlsym(handle.get(), dspqueue_read_pfn_t, dspqueue_read_pfn, dspqueue_read, பொய்);
    dlsym(handle.get(), dspqueue_read_noblock_pfn_t, dspqueue_read_noblock_pfn, dspqueue_read_noblock, பொய்);
    dlsym(handle.get(), remote_handle64_open_pfn_t, remote_handle64_open_pfn, remote_handle64_open, பொய்);
    dlsym(handle.get(), remote_handle64_invoke_pfn_t, remote_handle64_invoke_pfn, remote_handle64_invoke, பொய்);
    dlsym(handle.get(), remote_handle_control_pfn_t, remote_handle_control_pfn, remote_handle_control, பொய்);
    dlsym(handle.get(), remote_handle64_control_pfn_t, remote_handle64_control_pfn, remote_handle64_control, பொய்);
    dlsym(handle.get(), remote_session_control_pfn_t, remote_session_control_pfn, remote_session_control, பொய்);
    dlsym(handle.get(), remote_handle64_close_pfn_t, remote_handle64_close_pfn, remote_handle64_close, பொய்);
    dlsym(handle.get(), remote_system_request_pfn_t, remote_system_request_pfn, remote_system_request, உண்மை);

    lib_cdsp_rpc_handle = std::move(handle);
    initialized         = உண்மை;

    திரும்பு AEE_SUCCESS;
}

domain * htpdrv_get_domain(முழு domain_id) {
    முழு i    = 0;
    முழு size = அளவு(supported_domains) / அளவு(domain);

    க்கு (i = 0; i < size; i++) {
        எனில் (supported_domains[i].id == domain_id) {
            திரும்பு &supported_domains[i];
        }
    }

    திரும்பு வெற்று;
}

முழு htpdrv_get_arch(முழு domain, முழு * arch) {
    எனில் (!remote_handle_control_pfn) {
        GGML_LOG_ERROR("ggml-hex: remote_handle_control is not supported on this device\n");
        திரும்பு AEE_EUNSUPPORTEDAPI;
    }

    கட்டமைப்பு remote_dsp_capability arch_ver;
    arch_ver.domain       = (uint32_t) domain;
    arch_ver.attribute_ID = ARCH_VER;
    arch_ver.capability   = (uint32_t) 0;

    முழு err = remote_handle_control(DSPRPC_GET_DSP_INFO, &arch_ver, அளவு(arch_ver));
    எனில் ((err & 0xff) == (AEE_EUNSUPPORTEDAPI & 0xff)) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability API is not supported on this device\n");
        திரும்பு AEE_EUNSUPPORTEDAPI;
    }

    எனில் (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability query failed (err %d)\n", err);
        திரும்பு err;
    }

    uint32_t val = arch_ver.capability & 0xff;
    *arch = (முழு) ((val >> 4) * 10 + (val & 0x0f));
    திரும்பு 0;
}
