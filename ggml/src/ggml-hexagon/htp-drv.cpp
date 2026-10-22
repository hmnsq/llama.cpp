#ಅನ್ತರ್ಭಾವ <filesystem>
#ಅನ್ತರ್ಭಾವ <set>
#ಅನ್ತರ್ಭಾವ <sstream>
#ಅನ್ತರ್ಭಾವ <string>

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#ಅನ್ಯಥಾ
#   include <dlfcn.h>
#   include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"
#pragma clang diagnostic ignored "-Wnested-anon-types"

#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "htp-drv.h"
#ಅನ್ತರ್ಭಾವ "libdl.h"

#ಅನ್ತರ್ಭಾವ <domain.h>

//
// Driver API types
//

ಪ್ರಕಾರನಿರ್ದೇಶ ಶೂನ್ಯ * (*rpcmem_alloc_pfn_t)(ಪೂರ್ಣಾಙ್ಕ heapid, uint32_t flags, ಪೂರ್ಣಾಙ್ಕ size);
ಪ್ರಕಾರನಿರ್ದೇಶ ಶೂನ್ಯ * (*rpcmem_alloc2_pfn_t)(ಪೂರ್ಣಾಙ್ಕ heapid, uint32_t flags, size_t size);
ಪ್ರಕಾರನಿರ್ದೇಶ ಶೂನ್ಯ   (*rpcmem_free_pfn_t)(ಶೂನ್ಯ * po);
ಪ್ರಕಾರನಿರ್ದೇಶ ಪೂರ್ಣಾಙ್ಕ    (*rpcmem_to_fd_pfn_t)(ಶೂನ್ಯ * po);

ಪ್ರಕಾರನಿರ್ದೇಶ AEEResult (*dspqueue_create_pfn_t)(ಪೂರ್ಣಾಙ್ಕ                 domain,
                                           uint32_t            flags,
                                           uint32_t            req_queue_size,
                                           uint32_t            resp_queue_size,
                                           dspqueue_callback_t packet_callback,
                                           dspqueue_callback_t error_callback,
                                           ಶೂನ್ಯ *              callback_context,
                                           dspqueue_t *        queue);
ಪ್ರಕಾರನಿರ್ದೇಶ AEEResult (*dspqueue_close_pfn_t)(dspqueue_t queue);
ಪ್ರಕಾರನಿರ್ದೇಶ AEEResult (*dspqueue_export_pfn_t)(dspqueue_t queue, uint64_t *queue_id);
ಪ್ರಕಾರನಿರ್ದೇಶ AEEResult (*dspqueue_write_pfn_t)(dspqueue_t queue, uint32_t flags,
                                          uint32_t ಅಂಕೆ_buffers,
                                          ಸಂರಚನಾ dspqueue_buffer *buffers,
                                          uint32_t message_length,
                                          ಧ್ರುವ uint8_t *message,
                                          uint32_t timeout_us);
ಪ್ರಕಾರನಿರ್ದೇಶ AEEResult (*dspqueue_read_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                         uint32_t max_buffers, uint32_t *ಅಂಕೆ_buffers,
                                         ಸಂರಚನಾ dspqueue_buffer *buffers,
                                         uint32_t max_message_length,
                                         uint32_t *message_length, uint8_t *message,
                                         uint32_t timeout_us);
ಪ್ರಕಾರನಿರ್ದೇಶ AEEResult (*dspqueue_read_noblock_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                                 uint32_t max_buffers, uint32_t *ಅಂಕೆ_buffers,
                                                 ಸಂರಚನಾ dspqueue_buffer *buffers,
                                                 uint32_t max_message_length,
                                                 uint32_t *message_length, uint8_t *message);
ಪ್ರಕಾರನಿರ್ದೇಶ ಪೂರ್ಣಾಙ್ಕ (*fastrpc_mmap_pfn_t)(ಪೂರ್ಣಾಙ್ಕ domain, ಪೂರ್ಣಾಙ್ಕ fd, ಶೂನ್ಯ *addr, ಪೂರ್ಣಾಙ್ಕ offset, size_t length, ಗಣನಾ fastrpc_map_flags flags);
ಪ್ರಕಾರನಿರ್ದೇಶ ಪೂರ್ಣಾಙ್ಕ (*fastrpc_munmap_pfn_t)(ಪೂರ್ಣಾಙ್ಕ domain, ಪೂರ್ಣಾಙ್ಕ fd, ಶೂನ್ಯ *addr, size_t length);

ಪ್ರಕಾರನಿರ್ದೇಶ ಪೂರ್ಣಾಙ್ಕ (*remote_handle64_open_pfn_t)(ಧ್ರುವ ವರ್ಣ* name, remote_handle64 *ph);
ಪ್ರಕಾರನಿರ್ದೇಶ ಪೂರ್ಣಾಙ್ಕ (*remote_handle64_invoke_pfn_t)(remote_handle64 h, uint32_t dwScalars, remote_arg *pra);
ಪ್ರಕಾರನಿರ್ದೇಶ ಪೂರ್ಣಾಙ್ಕ (*remote_handle64_close_pfn_t)(remote_handle h);
ಪ್ರಕಾರನಿರ್ದೇಶ ಪೂರ್ಣಾಙ್ಕ (*remote_handle_control_pfn_t)(uint32_t req, ಶೂನ್ಯ* data, uint32_t datalen);
ಪ್ರಕಾರನಿರ್ದೇಶ ಪೂರ್ಣಾಙ್ಕ (*remote_handle64_control_pfn_t)(remote_handle64 h, uint32_t req, ಶೂನ್ಯ* data, uint32_t datalen);
ಪ್ರಕಾರನಿರ್ದೇಶ ಪೂರ್ಣಾಙ್ಕ (*remote_session_control_pfn_t)(uint32_t req, ಶೂನ್ಯ *data, uint32_t datalen);
ಪ್ರಕಾರನಿರ್ದೇಶ ಪೂರ್ಣಾಙ್ಕ (*remote_system_request_pfn_t)(system_req_payload * req);

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

ಶೂನ್ಯ * rpcmem_alloc(ಪೂರ್ಣಾಙ್ಕ heapid, uint32_t flags, ಪೂರ್ಣಾಙ್ಕ size) {
    ಪ್ರತ್ಯಾಗಮ rpcmem_alloc_pfn(heapid, flags, size);
}

ಶೂನ್ಯ * rpcmem_alloc2(ಪೂರ್ಣಾಙ್ಕ heapid, uint32_t flags, size_t size) {
    ಯದಿ (rpcmem_alloc2_pfn) {
        ಪ್ರತ್ಯಾಗಮ rpcmem_alloc2_pfn(heapid, flags, size);
    } ಅನ್ಯಥಾ {
        GGML_LOG_INFO("ggml-hex: rpcmem_alloc2 not found, falling back to rpcmem_alloc\n");
        ಪ್ರತ್ಯಾಗಮ rpcmem_alloc_pfn(heapid, flags, size);
    }
}

ಶೂನ್ಯ rpcmem_free(ಶೂನ್ಯ * po) {
    ಪ್ರತ್ಯಾಗಮ rpcmem_free_pfn(po);
}

ಪೂರ್ಣಾಙ್ಕ rpcmem_to_fd(ಶೂನ್ಯ * po) {
    ಪ್ರತ್ಯಾಗಮ rpcmem_to_fd_pfn(po);
}

HTPDRV_API ಪೂರ್ಣಾಙ್ಕ fastrpc_mmap(ಪೂರ್ಣಾಙ್ಕ domain, ಪೂರ್ಣಾಙ್ಕ fd, ಶೂನ್ಯ * addr, ಪೂರ್ಣಾಙ್ಕ offset, size_t length, ಗಣನಾ fastrpc_map_flags flags) {
    ಪ್ರತ್ಯಾಗಮ fastrpc_mmap_pfn(domain, fd, addr, offset, length, flags);
}

HTPDRV_API ಪೂರ್ಣಾಙ್ಕ fastrpc_munmap(ಪೂರ್ಣಾಙ್ಕ domain, ಪೂರ್ಣಾಙ್ಕ fd, ಶೂನ್ಯ * addr, size_t length) {
    ಪ್ರತ್ಯಾಗಮ fastrpc_munmap_pfn(domain, fd, addr, length);
}

AEEResult dspqueue_create(ಪೂರ್ಣಾಙ್ಕ                 domain,
                          uint32_t            flags,
                          uint32_t            req_queue_size,
                          uint32_t            resp_queue_size,
                          dspqueue_callback_t packet_callback,
                          dspqueue_callback_t error_callback,
                          ಶೂನ್ಯ *              callback_context,
                          dspqueue_t *        queue) {
    ಪ್ರತ್ಯಾಗಮ dspqueue_create_pfn(domain, flags, req_queue_size, resp_queue_size, packet_callback, error_callback,
                               callback_context, queue);
}

AEEResult dspqueue_close(dspqueue_t queue) {
    ಪ್ರತ್ಯಾಗಮ dspqueue_close_pfn(queue);
}

AEEResult dspqueue_export(dspqueue_t queue, uint64_t * queue_id) {
    ಪ್ರತ್ಯಾಗಮ dspqueue_export_pfn(queue, queue_id);
}

AEEResult dspqueue_write(dspqueue_t               queue,
                         uint32_t                 flags,
                         uint32_t                 ಅಂಕೆ_buffers,
                         ಸಂರಚನಾ dspqueue_buffer * buffers,
                         uint32_t                 message_length,
                         ಧ್ರುವ uint8_t *          message,
                         uint32_t                 timeout_us) {
    ಪ್ರತ್ಯಾಗಮ dspqueue_write_pfn(queue, flags, ಅಂಕೆ_buffers, buffers, message_length, message, timeout_us);
}

AEEResult dspqueue_read(dspqueue_t               queue,
                        uint32_t *               flags,
                        uint32_t                 max_buffers,
                        uint32_t *               ಅಂಕೆ_buffers,
                        ಸಂರಚನಾ dspqueue_buffer * buffers,
                        uint32_t                 max_message_length,
                        uint32_t *               message_length,
                        uint8_t *                message,
                        uint32_t                 timeout_us) {
#ifdef _WIN32
    ಯದಿ (timeout_us == 0) {
        ಪ್ರತ್ಯಾಗಮ dspqueue_read_noblock_pfn(queue, flags, max_buffers, ಅಂಕೆ_buffers, buffers, max_message_length,
                                     message_length, message);
    }
#endif
    ಪ್ರತ್ಯಾಗಮ dspqueue_read_pfn(queue, flags, max_buffers, ಅಂಕೆ_buffers, buffers, max_message_length, message_length,
                             message, timeout_us);
}

HTPDRV_API ಪೂರ್ಣಾಙ್ಕ remote_handle64_open(ಧ್ರುವ ವರ್ಣ * name, remote_handle64 * ph) {
    ಪ್ರತ್ಯಾಗಮ remote_handle64_open_pfn(name, ph);
}

HTPDRV_API ಪೂರ್ಣಾಙ್ಕ remote_handle64_invoke(remote_handle64 h, uint32_t dwScalars, remote_arg * pra) {
    ಪ್ರತ್ಯಾಗಮ remote_handle64_invoke_pfn(h, dwScalars, pra);
}

HTPDRV_API ಪೂರ್ಣಾಙ್ಕ remote_handle64_close(remote_handle64 h) {
    ಪ್ರತ್ಯಾಗಮ remote_handle64_close_pfn(h);
}

HTPDRV_API ಪೂರ್ಣಾಙ್ಕ remote_handle_control(uint32_t req, ಶೂನ್ಯ * data, uint32_t datalen) {
    ಪ್ರತ್ಯಾಗಮ remote_handle_control_pfn(req, data, datalen);
}

HTPDRV_API ಪೂರ್ಣಾಙ್ಕ remote_handle64_control(remote_handle64 h, uint32_t req, ಶೂನ್ಯ * data, uint32_t datalen) {
    ಪ್ರತ್ಯಾಗಮ remote_handle64_control_pfn(h, req, data, datalen);
}

HTPDRV_API ಪೂರ್ಣಾಙ್ಕ remote_session_control(uint32_t req, ಶೂನ್ಯ * data, uint32_t datalen) {
    ಪ್ರತ್ಯಾಗಮ remote_session_control_pfn(req, data, datalen);
}

HTPDRV_API ಪೂರ್ಣಾಙ್ಕ remote_system_request(system_req_payload * req) {
    ಯದಿ (!remote_system_request_pfn) {
        ಪ್ರತ್ಯಾಗಮ AEE_EUNSUPPORTEDAPI;
    }
    ಪ್ರತ್ಯಾಗಮ remote_system_request_pfn(req);
}

#ifdef _WIN32

ಸ್ಥಿರ std::string wstr_to_str(std::wstring_view wstr) {
    std::string result;
    ಯದಿ (wstr.empty()) {
        ಪ್ರತ್ಯಾಗಮ result;
    }
    auto bytes_needed = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.data(), (ಪೂರ್ಣಾಙ್ಕ) wstr.size(),
                                            nullptr, 0, nullptr, nullptr);
    ಯದಿ (bytes_needed == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        ಕ್ಷಿಪ std::runtime_error("Invalid wstring input");
    }

    result.resize(bytes_needed, '\0');
    ಪೂರ್ಣಾಙ್ಕ bytes_written = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.data(), (ಪೂರ್ಣಾಙ್ಕ) wstr.size(),
                                            result.data(), bytes_needed,
                                            nullptr, nullptr);
    ಯದಿ (bytes_written == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        ಕ್ಷಿಪ std::runtime_error("Wstring conversion failed");
    }
    ಪ್ರತ್ಯಾಗಮ result;
}

ಸ್ಥಿರ std::string get_driver_path() {
    std::wstring serviceName = L"qcnspmcdm";
    std::string result;

    // Get a handle to the SCM database.
    SC_HANDLE schSCManager = OpenSCManagerW(ರಿಕ್ತ, ರಿಕ್ತ, STANDARD_RIGHTS_READ);
    ಯದಿ (nullptr == schSCManager) {
        GGML_LOG_ERROR("ggml-hex: Failed to open SCManager. Error: %lu\n", GetLastError());
        ಪ್ರತ್ಯಾಗಮ result;
    }

    // Get a handle to the service.
    SC_HANDLE schService = OpenServiceW(schSCManager,           // SCM database
                                        serviceName.c_str(),    // name of service
                                        SERVICE_QUERY_CONFIG);  // need query config access

    ಯದಿ (nullptr == schService) {
        GGML_LOG_ERROR("ggml-hex: Failed to open qcnspmcdm service. Error: %lu\n", GetLastError());
        CloseServiceHandle(schSCManager);
        ಪ್ರತ್ಯಾಗಮ result;
    }

    // Store the size of buffer used as an output.
    DWORD bufferSize;
    ಯದಿ (!QueryServiceConfigW(schService, ರಿಕ್ತ, 0, &bufferSize) &&
        (GetLastError() != ERROR_INSUFFICIENT_BUFFER)) {
        GGML_LOG_ERROR("ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        ಪ್ರತ್ಯಾಗಮ result;
    }
    // Get the configuration of the service.
    LPQUERY_SERVICE_CONFIGW serviceConfig =
        static_cast<LPQUERY_SERVICE_CONFIGW>(LocalAlloc(LMEM_FIXED, bufferSize));
    ಯದಿ (!QueryServiceConfigW(schService, serviceConfig, bufferSize, &bufferSize)) {
        fprintf(stderr, "ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        LocalFree(serviceConfig);
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        ಪ್ರತ್ಯಾಗಮ result;
    }

    // Read the driver file path get its parent directory
    std::wstring driverPath = std::wstring(serviceConfig->lp_ದ್ವಿಮಾನ_Path_Name);
    driverPath = driverPath.substr(0, driverPath.find_last_of(L"\\"));

    // Clean up resources
    LocalFree(serviceConfig);
    CloseServiceHandle(schService);
    CloseServiceHandle(schSCManager);

    // Driver path would contain invalid path string, like:
    // \SystemRoot\System32\DriverStore\FileRepository\qcadsprpc8280.inf_arm64_c2b9460c9a072f37
    // "\SystemRoot" should be replace with a correct one (e.g. C:\Windows)
    ಧ್ರುವ std::wstring system_ಬೇರು_Placeholder = L"\\SystemRoot";
    ಯದಿ (0 != driverPath.compare(0, system_ಬೇರು_Placeholder.length(), system_ಬೇರು_Placeholder)) {
        GGML_LOG_ERROR("ggml-hex: String pattern not found in driver path.\n");
        ಪ್ರತ್ಯಾಗಮ result;
    }

    // Replace \SystemRoot with an absolute path from system ENV windir
    ಧ್ರುವ std::wstring system_ಬೇರು_Env = L"windir";

    // Query the number of wide characters this variable requires
    DWORD ಅಂಕೆ_ಪದಗಳು = GetEnvironmentVariableW(system_ಬೇರು_Env.c_str(), ರಿಕ್ತ, 0);
    ಯದಿ (ಅಂಕೆ_ಪದಗಳು == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed get systemRoot environment variable\n");
        ಪ್ರತ್ಯಾಗಮ result;
    }

    // Query the actual system root name from environment variable
    std::vector<wchar_t> system_ಬೇರು(ಅಂಕೆ_ಪದಗಳು + 1);
    ಅಂಕೆ_ಪದಗಳು = GetEnvironmentVariableW(system_ಬೇರು_Env.c_str(), system_ಬೇರು.data(), ಅಂಕೆ_ಪದಗಳು + 1);
    ಯದಿ (ಅಂಕೆ_ಪದಗಳು == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed to read windir environment variable\n");
        ಪ್ರತ್ಯಾಗಮ result;
    }
    driverPath.replace(0, system_ಬೇರು_Placeholder.length(), std::wstring(system_ಬೇರು.data()));

    ಪ್ರತ್ಯಾಗಮ wstr_to_str(driverPath);
}

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

ಪೂರ್ಣಾಙ್ಕ htpdrv_init() {
    ಸ್ಥಿರ dl_handle_ptr lib_cdsp_rpc_handle = nullptr;
    ಸ್ಥಿರ ತಾರ್ಕಿಕ initialized = ಅಸತ್ಯ;
#ifdef _WIN32
    std::string drv_path = get_driver_path() + "\\" + "libcdsprpc.dll";
#ಅನ್ಯಥಾ
    std::string drv_path = "libcdsprpc.so";
#endif
    ಯದಿ (initialized) {
        GGML_LOG_INFO("ggml-hex: Driver already loaded\n");
        ಪ್ರತ್ಯಾಗಮ AEE_SUCCESS;
    }
    GGML_LOG_INFO("ggml-hex: Loading driver %s\n", drv_path.c_str());

    fs::path path{ drv_path.c_str() };
    dl_handle_ptr handle { dl_load_library(path) };
    ಯದಿ (!handle) {
        GGML_LOG_ERROR("ggml-hex: failed to load %s: %s\n", path.u8string().c_str(), dl_error());
        ಪ್ರತ್ಯಾಗಮ AEE_EUNABLETOLOAD;
    }

#define dlsym(drv, type, pfn, ಚಿಹ್ನೆ, ignore)                               \
    ಕುರು {                                                                    \
        pfn = (type) dl_get_sym(drv, #symbol);                              \
        ಯದಿ (!ignore && nullptr == pfn) {                                    \
            GGML_LOG_ERROR("ggml-hex: failed to dlsym %s\n", #symbol);      \
            ಪ್ರತ್ಯಾಗಮ AEE_EUNABLETOLOAD;                                       \
        }                                                                   \
    } ಯಾವತ್ (0)

    dlsym(handle.get(), rpcmem_alloc_pfn_t, rpcmem_alloc_pfn, rpcmem_alloc, ಅಸತ್ಯ);
    dlsym(handle.get(), rpcmem_alloc2_pfn_t, rpcmem_alloc2_pfn, rpcmem_alloc2, ಸತ್ಯ);
    dlsym(handle.get(), rpcmem_free_pfn_t, rpcmem_free_pfn, rpcmem_free, ಅಸತ್ಯ);
    dlsym(handle.get(), rpcmem_to_fd_pfn_t, rpcmem_to_fd_pfn, rpcmem_to_fd, ಅಸತ್ಯ);
    dlsym(handle.get(), fastrpc_mmap_pfn_t, fastrpc_mmap_pfn, fastrpc_mmap, ಅಸತ್ಯ);
    dlsym(handle.get(), fastrpc_munmap_pfn_t, fastrpc_munmap_pfn, fastrpc_munmap, ಅಸತ್ಯ);
    dlsym(handle.get(), dspqueue_create_pfn_t, dspqueue_create_pfn, dspqueue_create, ಅಸತ್ಯ);
    dlsym(handle.get(), dspqueue_close_pfn_t, dspqueue_close_pfn, dspqueue_close, ಅಸತ್ಯ);
    dlsym(handle.get(), dspqueue_export_pfn_t, dspqueue_export_pfn, dspqueue_export, ಅಸತ್ಯ);
    dlsym(handle.get(), dspqueue_write_pfn_t, dspqueue_write_pfn, dspqueue_write, ಅಸತ್ಯ);
    dlsym(handle.get(), dspqueue_read_pfn_t, dspqueue_read_pfn, dspqueue_read, ಅಸತ್ಯ);
    dlsym(handle.get(), dspqueue_read_noblock_pfn_t, dspqueue_read_noblock_pfn, dspqueue_read_noblock, ಅಸತ್ಯ);
    dlsym(handle.get(), remote_handle64_open_pfn_t, remote_handle64_open_pfn, remote_handle64_open, ಅಸತ್ಯ);
    dlsym(handle.get(), remote_handle64_invoke_pfn_t, remote_handle64_invoke_pfn, remote_handle64_invoke, ಅಸತ್ಯ);
    dlsym(handle.get(), remote_handle_control_pfn_t, remote_handle_control_pfn, remote_handle_control, ಅಸತ್ಯ);
    dlsym(handle.get(), remote_handle64_control_pfn_t, remote_handle64_control_pfn, remote_handle64_control, ಅಸತ್ಯ);
    dlsym(handle.get(), remote_session_control_pfn_t, remote_session_control_pfn, remote_session_control, ಅಸತ್ಯ);
    dlsym(handle.get(), remote_handle64_close_pfn_t, remote_handle64_close_pfn, remote_handle64_close, ಅಸತ್ಯ);
    dlsym(handle.get(), remote_system_request_pfn_t, remote_system_request_pfn, remote_system_request, ಸತ್ಯ);

    lib_cdsp_rpc_handle = std::move(handle);
    initialized         = ಸತ್ಯ;

    ಪ್ರತ್ಯಾಗಮ AEE_SUCCESS;
}

domain * htpdrv_get_domain(ಪೂರ್ಣಾಙ್ಕ domain_id) {
    ಪೂರ್ಣಾಙ್ಕ i    = 0;
    ಪೂರ್ಣಾಙ್ಕ size = ಪರಿಮಾಣ(supported_domains) / ಪರಿಮಾಣ(domain);

    ಪ್ರತಿ (i = 0; i < size; i++) {
        ಯದಿ (supported_domains[i].id == domain_id) {
            ಪ್ರತ್ಯಾಗಮ &supported_domains[i];
        }
    }

    ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
}

ಪೂರ್ಣಾಙ್ಕ htpdrv_get_arch(ಪೂರ್ಣಾಙ್ಕ domain, ಪೂರ್ಣಾಙ್ಕ * arch) {
    ಯದಿ (!remote_handle_control_pfn) {
        GGML_LOG_ERROR("ggml-hex: remote_handle_control is not supported on this device\n");
        ಪ್ರತ್ಯಾಗಮ AEE_EUNSUPPORTEDAPI;
    }

    ಸಂರಚನಾ remote_dsp_capability arch_ver;
    arch_ver.domain       = (uint32_t) domain;
    arch_ver.attribute_ID = ARCH_VER;
    arch_ver.capability   = (uint32_t) 0;

    ಪೂರ್ಣಾಙ್ಕ err = remote_handle_control(DSPRPC_GET_DSP_INFO, &arch_ver, ಪರಿಮಾಣ(arch_ver));
    ಯದಿ ((err & 0xff) == (AEE_EUNSUPPORTEDAPI & 0xff)) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability API is not supported on this device\n");
        ಪ್ರತ್ಯಾಗಮ AEE_EUNSUPPORTEDAPI;
    }

    ಯದಿ (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability query failed (err %d)\n", err);
        ಪ್ರತ್ಯಾಗಮ err;
    }

    uint32_t val = arch_ver.capability & 0xff;
    *arch = (ಪೂರ್ಣಾಙ್ಕ) ((val >> 4) * 10 + (val & 0x0f));
    ಪ್ರತ್ಯಾಗಮ 0;
}
