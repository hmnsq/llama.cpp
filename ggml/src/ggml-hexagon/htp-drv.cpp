#ଅନ୍ତର୍ଭାଵ <filesystem>
#ଅନ୍ତର୍ଭାଵ <set>
#ଅନ୍ତର୍ଭାଵ <sstream>
#ଅନ୍ତର୍ଭାଵ <string>

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#ଅନ୍ୟଥା
#   include <dlfcn.h>
#   include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"
#pragma clang diagnostic ignored "-Wnested-anon-types"

#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "htp-drv.h"
#ଅନ୍ତର୍ଭାଵ "libdl.h"

#ଅନ୍ତର୍ଭାଵ <domain.h>

//
// Driver API types
//

ପ୍ରକାରବ୍ୟାଖ୍ୟା ଶୂନ୍ୟ * (*rpcmem_alloc_pfn_t)(ପୂର୍ଣ୍ଣସଂଖ୍ୟା heapid, uint32_t flags, ପୂର୍ଣ୍ଣସଂଖ୍ୟା size);
ପ୍ରକାରବ୍ୟାଖ୍ୟା ଶୂନ୍ୟ * (*rpcmem_alloc2_pfn_t)(ପୂର୍ଣ୍ଣସଂଖ୍ୟା heapid, uint32_t flags, size_t size);
ପ୍ରକାରବ୍ୟାଖ୍ୟା ଶୂନ୍ୟ   (*rpcmem_free_pfn_t)(ଶୂନ୍ୟ * po);
ପ୍ରକାରବ୍ୟାଖ୍ୟା ପୂର୍ଣ୍ଣସଂଖ୍ୟା    (*rpcmem_to_fd_pfn_t)(ଶୂନ୍ୟ * po);

ପ୍ରକାରବ୍ୟାଖ୍ୟା AEEResult (*dspqueue_create_pfn_t)(ପୂର୍ଣ୍ଣସଂଖ୍ୟା                 domain,
                                           uint32_t            flags,
                                           uint32_t            req_queue_size,
                                           uint32_t            resp_queue_size,
                                           dspqueue_callback_t packet_callback,
                                           dspqueue_callback_t error_callback,
                                           ଶୂନ୍ୟ *              callback_context,
                                           dspqueue_t *        queue);
ପ୍ରକାରବ୍ୟାଖ୍ୟା AEEResult (*dspqueue_close_pfn_t)(dspqueue_t queue);
ପ୍ରକାରବ୍ୟାଖ୍ୟା AEEResult (*dspqueue_export_pfn_t)(dspqueue_t queue, uint64_t *queue_id);
ପ୍ରକାରବ୍ୟାଖ୍ୟା AEEResult (*dspqueue_write_pfn_t)(dspqueue_t queue, uint32_t flags,
                                          uint32_t ଅଙ୍କ_buffers,
                                          ସଂରଚନା dspqueue_buffer *buffers,
                                          uint32_t message_length,
                                          ଅଚଳ uint8_t *message,
                                          uint32_t timeout_us);
ପ୍ରକାରବ୍ୟାଖ୍ୟା AEEResult (*dspqueue_read_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                         uint32_t max_buffers, uint32_t *ଅଙ୍କ_buffers,
                                         ସଂରଚନା dspqueue_buffer *buffers,
                                         uint32_t max_message_length,
                                         uint32_t *message_length, uint8_t *message,
                                         uint32_t timeout_us);
ପ୍ରକାରବ୍ୟାଖ୍ୟା AEEResult (*dspqueue_read_noblock_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                                 uint32_t max_buffers, uint32_t *ଅଙ୍କ_buffers,
                                                 ସଂରଚନା dspqueue_buffer *buffers,
                                                 uint32_t max_message_length,
                                                 uint32_t *message_length, uint8_t *message);
ପ୍ରକାରବ୍ୟାଖ୍ୟା ପୂର୍ଣ୍ଣସଂଖ୍ୟା (*fastrpc_mmap_pfn_t)(ପୂର୍ଣ୍ଣସଂଖ୍ୟା domain, ପୂର୍ଣ୍ଣସଂଖ୍ୟା fd, ଶୂନ୍ୟ *addr, ପୂର୍ଣ୍ଣସଂଖ୍ୟା offset, size_t length, ଗଣନା fastrpc_map_flags flags);
ପ୍ରକାରବ୍ୟାଖ୍ୟା ପୂର୍ଣ୍ଣସଂଖ୍ୟା (*fastrpc_munmap_pfn_t)(ପୂର୍ଣ୍ଣସଂଖ୍ୟା domain, ପୂର୍ଣ୍ଣସଂଖ୍ୟା fd, ଶୂନ୍ୟ *addr, size_t length);

ପ୍ରକାରବ୍ୟାଖ୍ୟା ପୂର୍ଣ୍ଣସଂଖ୍ୟା (*remote_handle64_open_pfn_t)(ଅଚଳ ଅକ୍ଷର* name, remote_handle64 *ph);
ପ୍ରକାରବ୍ୟାଖ୍ୟା ପୂର୍ଣ୍ଣସଂଖ୍ୟା (*remote_handle64_invoke_pfn_t)(remote_handle64 h, uint32_t dwScalars, remote_arg *pra);
ପ୍ରକାରବ୍ୟାଖ୍ୟା ପୂର୍ଣ୍ଣସଂଖ୍ୟା (*remote_handle64_close_pfn_t)(remote_handle h);
ପ୍ରକାରବ୍ୟାଖ୍ୟା ପୂର୍ଣ୍ଣସଂଖ୍ୟା (*remote_handle_control_pfn_t)(uint32_t req, ଶୂନ୍ୟ* data, uint32_t datalen);
ପ୍ରକାରବ୍ୟାଖ୍ୟା ପୂର୍ଣ୍ଣସଂଖ୍ୟା (*remote_handle64_control_pfn_t)(remote_handle64 h, uint32_t req, ଶୂନ୍ୟ* data, uint32_t datalen);
ପ୍ରକାରବ୍ୟାଖ୍ୟା ପୂର୍ଣ୍ଣସଂଖ୍ୟା (*remote_session_control_pfn_t)(uint32_t req, ଶୂନ୍ୟ *data, uint32_t datalen);
ପ୍ରକାରବ୍ୟାଖ୍ୟା ପୂର୍ଣ୍ଣସଂଖ୍ୟା (*remote_system_request_pfn_t)(system_req_payload * req);

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

ଶୂନ୍ୟ * rpcmem_alloc(ପୂର୍ଣ୍ଣସଂଖ୍ୟା heapid, uint32_t flags, ପୂର୍ଣ୍ଣସଂଖ୍ୟା size) {
    ଫେରନ୍ତୁ rpcmem_alloc_pfn(heapid, flags, size);
}

ଶୂନ୍ୟ * rpcmem_alloc2(ପୂର୍ଣ୍ଣସଂଖ୍ୟା heapid, uint32_t flags, size_t size) {
    ଯଦି (rpcmem_alloc2_pfn) {
        ଫେରନ୍ତୁ rpcmem_alloc2_pfn(heapid, flags, size);
    } ଅନ୍ୟଥା {
        GGML_LOG_INFO("ggml-hex: rpcmem_alloc2 not found, falling back to rpcmem_alloc\n");
        ଫେରନ୍ତୁ rpcmem_alloc_pfn(heapid, flags, size);
    }
}

ଶୂନ୍ୟ rpcmem_free(ଶୂନ୍ୟ * po) {
    ଫେରନ୍ତୁ rpcmem_free_pfn(po);
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା rpcmem_to_fd(ଶୂନ୍ୟ * po) {
    ଫେରନ୍ତୁ rpcmem_to_fd_pfn(po);
}

HTPDRV_API ପୂର୍ଣ୍ଣସଂଖ୍ୟା fastrpc_mmap(ପୂର୍ଣ୍ଣସଂଖ୍ୟା domain, ପୂର୍ଣ୍ଣସଂଖ୍ୟା fd, ଶୂନ୍ୟ * addr, ପୂର୍ଣ୍ଣସଂଖ୍ୟା offset, size_t length, ଗଣନା fastrpc_map_flags flags) {
    ଫେରନ୍ତୁ fastrpc_mmap_pfn(domain, fd, addr, offset, length, flags);
}

HTPDRV_API ପୂର୍ଣ୍ଣସଂଖ୍ୟା fastrpc_munmap(ପୂର୍ଣ୍ଣସଂଖ୍ୟା domain, ପୂର୍ଣ୍ଣସଂଖ୍ୟା fd, ଶୂନ୍ୟ * addr, size_t length) {
    ଫେରନ୍ତୁ fastrpc_munmap_pfn(domain, fd, addr, length);
}

AEEResult dspqueue_create(ପୂର୍ଣ୍ଣସଂଖ୍ୟା                 domain,
                          uint32_t            flags,
                          uint32_t            req_queue_size,
                          uint32_t            resp_queue_size,
                          dspqueue_callback_t packet_callback,
                          dspqueue_callback_t error_callback,
                          ଶୂନ୍ୟ *              callback_context,
                          dspqueue_t *        queue) {
    ଫେରନ୍ତୁ dspqueue_create_pfn(domain, flags, req_queue_size, resp_queue_size, packet_callback, error_callback,
                               callback_context, queue);
}

AEEResult dspqueue_close(dspqueue_t queue) {
    ଫେରନ୍ତୁ dspqueue_close_pfn(queue);
}

AEEResult dspqueue_export(dspqueue_t queue, uint64_t * queue_id) {
    ଫେରନ୍ତୁ dspqueue_export_pfn(queue, queue_id);
}

AEEResult dspqueue_write(dspqueue_t               queue,
                         uint32_t                 flags,
                         uint32_t                 ଅଙ୍କ_buffers,
                         ସଂରଚନା dspqueue_buffer * buffers,
                         uint32_t                 message_length,
                         ଅଚଳ uint8_t *          message,
                         uint32_t                 timeout_us) {
    ଫେରନ୍ତୁ dspqueue_write_pfn(queue, flags, ଅଙ୍କ_buffers, buffers, message_length, message, timeout_us);
}

AEEResult dspqueue_read(dspqueue_t               queue,
                        uint32_t *               flags,
                        uint32_t                 max_buffers,
                        uint32_t *               ଅଙ୍କ_buffers,
                        ସଂରଚନା dspqueue_buffer * buffers,
                        uint32_t                 max_message_length,
                        uint32_t *               message_length,
                        uint8_t *                message,
                        uint32_t                 timeout_us) {
#ifdef _WIN32
    ଯଦି (timeout_us == 0) {
        ଫେରନ୍ତୁ dspqueue_read_noblock_pfn(queue, flags, max_buffers, ଅଙ୍କ_buffers, buffers, max_message_length,
                                     message_length, message);
    }
#endif
    ଫେରନ୍ତୁ dspqueue_read_pfn(queue, flags, max_buffers, ଅଙ୍କ_buffers, buffers, max_message_length, message_length,
                             message, timeout_us);
}

HTPDRV_API ପୂର୍ଣ୍ଣସଂଖ୍ୟା remote_handle64_open(ଅଚଳ ଅକ୍ଷର * name, remote_handle64 * ph) {
    ଫେରନ୍ତୁ remote_handle64_open_pfn(name, ph);
}

HTPDRV_API ପୂର୍ଣ୍ଣସଂଖ୍ୟା remote_handle64_invoke(remote_handle64 h, uint32_t dwScalars, remote_arg * pra) {
    ଫେରନ୍ତୁ remote_handle64_invoke_pfn(h, dwScalars, pra);
}

HTPDRV_API ପୂର୍ଣ୍ଣସଂଖ୍ୟା remote_handle64_close(remote_handle64 h) {
    ଫେରନ୍ତୁ remote_handle64_close_pfn(h);
}

HTPDRV_API ପୂର୍ଣ୍ଣସଂଖ୍ୟା remote_handle_control(uint32_t req, ଶୂନ୍ୟ * data, uint32_t datalen) {
    ଫେରନ୍ତୁ remote_handle_control_pfn(req, data, datalen);
}

HTPDRV_API ପୂର୍ଣ୍ଣସଂଖ୍ୟା remote_handle64_control(remote_handle64 h, uint32_t req, ଶୂନ୍ୟ * data, uint32_t datalen) {
    ଫେରନ୍ତୁ remote_handle64_control_pfn(h, req, data, datalen);
}

HTPDRV_API ପୂର୍ଣ୍ଣସଂଖ୍ୟା remote_session_control(uint32_t req, ଶୂନ୍ୟ * data, uint32_t datalen) {
    ଫେରନ୍ତୁ remote_session_control_pfn(req, data, datalen);
}

HTPDRV_API ପୂର୍ଣ୍ଣସଂଖ୍ୟା remote_system_request(system_req_payload * req) {
    ଯଦି (!remote_system_request_pfn) {
        ଫେରନ୍ତୁ AEE_EUNSUPPORTEDAPI;
    }
    ଫେରନ୍ତୁ remote_system_request_pfn(req);
}

#ifdef _WIN32

ସ୍ଥିର std::string wstr_to_str(std::wstring_view wstr) {
    std::string result;
    ଯଦି (wstr.empty()) {
        ଫେରନ୍ତୁ result;
    }
    auto bytes_needed = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.data(), (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) wstr.size(),
                                            nullptr, 0, nullptr, nullptr);
    ଯଦି (bytes_needed == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        ଫିଙ୍ଗ std::runtime_error("Invalid wstring input");
    }

    result.resize(bytes_needed, '\0');
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା bytes_written = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.data(), (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) wstr.size(),
                                            result.data(), bytes_needed,
                                            nullptr, nullptr);
    ଯଦି (bytes_written == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        ଫିଙ୍ଗ std::runtime_error("Wstring conversion failed");
    }
    ଫେରନ୍ତୁ result;
}

ସ୍ଥିର std::string get_driver_path() {
    std::wstring serviceName = L"qcnspmcdm";
    std::string result;

    // Get a handle to the SCM database.
    SC_HANDLE schSCManager = OpenSCManagerW(ଖାଲି, ଖାଲି, STANDARD_RIGHTS_READ);
    ଯଦି (nullptr == schSCManager) {
        GGML_LOG_ERROR("ggml-hex: Failed to open SCManager. Error: %lu\n", GetLastError());
        ଫେରନ୍ତୁ result;
    }

    // Get a handle to the service.
    SC_HANDLE schService = OpenServiceW(schSCManager,           // SCM database
                                        serviceName.c_str(),    // name of service
                                        SERVICE_QUERY_CONFIG);  // need query config access

    ଯଦି (nullptr == schService) {
        GGML_LOG_ERROR("ggml-hex: Failed to open qcnspmcdm service. Error: %lu\n", GetLastError());
        CloseServiceHandle(schSCManager);
        ଫେରନ୍ତୁ result;
    }

    // Store the size of buffer used as an output.
    DWORD bufferSize;
    ଯଦି (!QueryServiceConfigW(schService, ଖାଲି, 0, &bufferSize) &&
        (GetLastError() != ERROR_INSUFFICIENT_BUFFER)) {
        GGML_LOG_ERROR("ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        ଫେରନ୍ତୁ result;
    }
    // Get the configuration of the service.
    LPQUERY_SERVICE_CONFIGW serviceConfig =
        static_cast<LPQUERY_SERVICE_CONFIGW>(LocalAlloc(LMEM_FIXED, bufferSize));
    ଯଦି (!QueryServiceConfigW(schService, serviceConfig, bufferSize, &bufferSize)) {
        fprintf(stderr, "ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        LocalFree(serviceConfig);
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        ଫେରନ୍ତୁ result;
    }

    // Read the driver file path get its parent directory
    std::wstring driverPath = std::wstring(serviceConfig->lp_ଦ୍ୱିମିକ_Path_Name);
    driverPath = driverPath.substr(0, driverPath.find_last_of(L"\\"));

    // Clean up resources
    LocalFree(serviceConfig);
    CloseServiceHandle(schService);
    CloseServiceHandle(schSCManager);

    // Driver path would contain invalid path string, like:
    // \SystemRoot\System32\DriverStore\FileRepository\qcadsprpc8280.inf_arm64_c2b9460c9a072f37
    // "\SystemRoot" should be replace with a correct one (e.g. C:\Windows)
    ଅଚଳ std::wstring system_ମୂଳ_Placeholder = L"\\SystemRoot";
    ଯଦି (0 != driverPath.compare(0, system_ମୂଳ_Placeholder.length(), system_ମୂଳ_Placeholder)) {
        GGML_LOG_ERROR("ggml-hex: String pattern not found in driver path.\n");
        ଫେରନ୍ତୁ result;
    }

    // Replace \SystemRoot with an absolute path from system ENV windir
    ଅଚଳ std::wstring system_ମୂଳ_Env = L"windir";

    // Query the number of wide characters this variable requires
    DWORD ଅଙ୍କ_ଶବ୍ଦଗୁଡ଼ିକ = GetEnvironmentVariableW(system_ମୂଳ_Env.c_str(), ଖାଲି, 0);
    ଯଦି (ଅଙ୍କ_ଶବ୍ଦଗୁଡ଼ିକ == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed get systemRoot environment variable\n");
        ଫେରନ୍ତୁ result;
    }

    // Query the actual system root name from environment variable
    std::vector<wchar_t> system_ମୂଳ(ଅଙ୍କ_ଶବ୍ଦଗୁଡ଼ିକ + 1);
    ଅଙ୍କ_ଶବ୍ଦଗୁଡ଼ିକ = GetEnvironmentVariableW(system_ମୂଳ_Env.c_str(), system_ମୂଳ.data(), ଅଙ୍କ_ଶବ୍ଦଗୁଡ଼ିକ + 1);
    ଯଦି (ଅଙ୍କ_ଶବ୍ଦଗୁଡ଼ିକ == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed to read windir environment variable\n");
        ଫେରନ୍ତୁ result;
    }
    driverPath.replace(0, system_ମୂଳ_Placeholder.length(), std::wstring(system_ମୂଳ.data()));

    ଫେରନ୍ତୁ wstr_to_str(driverPath);
}

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

ପୂର୍ଣ୍ଣସଂଖ୍ୟା htpdrv_init() {
    ସ୍ଥିର dl_handle_ptr lib_cdsp_rpc_handle = nullptr;
    ସ୍ଥିର ତାର୍କିକ initialized = ମିଥ୍ୟା;
#ifdef _WIN32
    std::string drv_path = get_driver_path() + "\\" + "libcdsprpc.dll";
#ଅନ୍ୟଥା
    std::string drv_path = "libcdsprpc.so";
#endif
    ଯଦି (initialized) {
        GGML_LOG_INFO("ggml-hex: Driver already loaded\n");
        ଫେରନ୍ତୁ AEE_SUCCESS;
    }
    GGML_LOG_INFO("ggml-hex: Loading driver %s\n", drv_path.c_str());

    fs::path path{ drv_path.c_str() };
    dl_handle_ptr handle { dl_load_library(path) };
    ଯଦି (!handle) {
        GGML_LOG_ERROR("ggml-hex: failed to load %s: %s\n", path.u8string().c_str(), dl_error());
        ଫେରନ୍ତୁ AEE_EUNABLETOLOAD;
    }

#define dlsym(drv, type, pfn, ପ୍ରତୀକ, ignore)                               \
    କର {                                                                    \
        pfn = (type) dl_get_sym(drv, #symbol);                              \
        ଯଦି (!ignore && nullptr == pfn) {                                    \
            GGML_LOG_ERROR("ggml-hex: failed to dlsym %s\n", #symbol);      \
            ଫେରନ୍ତୁ AEE_EUNABLETOLOAD;                                       \
        }                                                                   \
    } ଯେତେବେଳେ (0)

    dlsym(handle.get(), rpcmem_alloc_pfn_t, rpcmem_alloc_pfn, rpcmem_alloc, ମିଥ୍ୟା);
    dlsym(handle.get(), rpcmem_alloc2_pfn_t, rpcmem_alloc2_pfn, rpcmem_alloc2, ସତ୍ୟ);
    dlsym(handle.get(), rpcmem_free_pfn_t, rpcmem_free_pfn, rpcmem_free, ମିଥ୍ୟା);
    dlsym(handle.get(), rpcmem_to_fd_pfn_t, rpcmem_to_fd_pfn, rpcmem_to_fd, ମିଥ୍ୟା);
    dlsym(handle.get(), fastrpc_mmap_pfn_t, fastrpc_mmap_pfn, fastrpc_mmap, ମିଥ୍ୟା);
    dlsym(handle.get(), fastrpc_munmap_pfn_t, fastrpc_munmap_pfn, fastrpc_munmap, ମିଥ୍ୟା);
    dlsym(handle.get(), dspqueue_create_pfn_t, dspqueue_create_pfn, dspqueue_create, ମିଥ୍ୟା);
    dlsym(handle.get(), dspqueue_close_pfn_t, dspqueue_close_pfn, dspqueue_close, ମିଥ୍ୟା);
    dlsym(handle.get(), dspqueue_export_pfn_t, dspqueue_export_pfn, dspqueue_export, ମିଥ୍ୟା);
    dlsym(handle.get(), dspqueue_write_pfn_t, dspqueue_write_pfn, dspqueue_write, ମିଥ୍ୟା);
    dlsym(handle.get(), dspqueue_read_pfn_t, dspqueue_read_pfn, dspqueue_read, ମିଥ୍ୟା);
    dlsym(handle.get(), dspqueue_read_noblock_pfn_t, dspqueue_read_noblock_pfn, dspqueue_read_noblock, ମିଥ୍ୟା);
    dlsym(handle.get(), remote_handle64_open_pfn_t, remote_handle64_open_pfn, remote_handle64_open, ମିଥ୍ୟା);
    dlsym(handle.get(), remote_handle64_invoke_pfn_t, remote_handle64_invoke_pfn, remote_handle64_invoke, ମିଥ୍ୟା);
    dlsym(handle.get(), remote_handle_control_pfn_t, remote_handle_control_pfn, remote_handle_control, ମିଥ୍ୟା);
    dlsym(handle.get(), remote_handle64_control_pfn_t, remote_handle64_control_pfn, remote_handle64_control, ମିଥ୍ୟା);
    dlsym(handle.get(), remote_session_control_pfn_t, remote_session_control_pfn, remote_session_control, ମିଥ୍ୟା);
    dlsym(handle.get(), remote_handle64_close_pfn_t, remote_handle64_close_pfn, remote_handle64_close, ମିଥ୍ୟା);
    dlsym(handle.get(), remote_system_request_pfn_t, remote_system_request_pfn, remote_system_request, ସତ୍ୟ);

    lib_cdsp_rpc_handle = std::move(handle);
    initialized         = ସତ୍ୟ;

    ଫେରନ୍ତୁ AEE_SUCCESS;
}

domain * htpdrv_get_domain(ପୂର୍ଣ୍ଣସଂଖ୍ୟା domain_id) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା i    = 0;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା size = ଆକାର(supported_domains) / ଆକାର(domain);

    ପାଇଁ (i = 0; i < size; i++) {
        ଯଦି (supported_domains[i].id == domain_id) {
            ଫେରନ୍ତୁ &supported_domains[i];
        }
    }

    ଫେରନ୍ତୁ ଖାଲି;
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା htpdrv_get_arch(ପୂର୍ଣ୍ଣସଂଖ୍ୟା domain, ପୂର୍ଣ୍ଣସଂଖ୍ୟା * arch) {
    ଯଦି (!remote_handle_control_pfn) {
        GGML_LOG_ERROR("ggml-hex: remote_handle_control is not supported on this device\n");
        ଫେରନ୍ତୁ AEE_EUNSUPPORTEDAPI;
    }

    ସଂରଚନା remote_dsp_capability arch_ver;
    arch_ver.domain       = (uint32_t) domain;
    arch_ver.attribute_ID = ARCH_VER;
    arch_ver.capability   = (uint32_t) 0;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା err = remote_handle_control(DSPRPC_GET_DSP_INFO, &arch_ver, ଆକାର(arch_ver));
    ଯଦି ((err & 0xff) == (AEE_EUNSUPPORTEDAPI & 0xff)) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability API is not supported on this device\n");
        ଫେରନ୍ତୁ AEE_EUNSUPPORTEDAPI;
    }

    ଯଦି (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability query failed (err %d)\n", err);
        ଫେରନ୍ତୁ err;
    }

    uint32_t val = arch_ver.capability & 0xff;
    *arch = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ((val >> 4) * 10 + (val & 0x0f));
    ଫେରନ୍ତୁ 0;
}
