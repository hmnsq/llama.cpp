#شامل <filesystem>
#شامل <set>
#شامل <sstream>
#شامل <string>

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#وگرنه
#   include <dlfcn.h>
#   include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"
#pragma clang diagnostic ignored "-Wnested-anon-types"

#شامل "ggml-impl.h"
#شامل "htp-drv.h"
#شامل "libdl.h"

#شامل <domain.h>

//
// Driver API types
//

تعریفنوع تهی * (*rpcmem_alloc_pfn_t)(صحیح heapid, uint32_t flags, صحیح size);
تعریفنوع تهی * (*rpcmem_alloc2_pfn_t)(صحیح heapid, uint32_t flags, size_t size);
تعریفنوع تهی   (*rpcmem_free_pfn_t)(تهی * po);
تعریفنوع صحیح    (*rpcmem_to_fd_pfn_t)(تهی * po);

تعریفنوع AEEResult (*dspqueue_create_pfn_t)(صحیح                 domain,
                                           uint32_t            flags,
                                           uint32_t            req_queue_size,
                                           uint32_t            resp_queue_size,
                                           dspqueue_callback_t packet_callback,
                                           dspqueue_callback_t error_callback,
                                           تهی *              callback_context,
                                           dspqueue_t *        queue);
تعریفنوع AEEResult (*dspqueue_close_pfn_t)(dspqueue_t queue);
تعریفنوع AEEResult (*dspqueue_export_pfn_t)(dspqueue_t queue, uint64_t *queue_id);
تعریفنوع AEEResult (*dspqueue_write_pfn_t)(dspqueue_t queue, uint32_t flags,
                                          uint32_t رقم_buffers,
                                          ساختار dspqueue_buffer *buffers,
                                          uint32_t message_length,
                                          ثابت uint8_t *message,
                                          uint32_t timeout_us);
تعریفنوع AEEResult (*dspqueue_read_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                         uint32_t max_buffers, uint32_t *رقم_buffers,
                                         ساختار dspqueue_buffer *buffers,
                                         uint32_t max_message_length,
                                         uint32_t *message_length, uint8_t *message,
                                         uint32_t timeout_us);
تعریفنوع AEEResult (*dspqueue_read_noblock_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                                 uint32_t max_buffers, uint32_t *رقم_buffers,
                                                 ساختار dspqueue_buffer *buffers,
                                                 uint32_t max_message_length,
                                                 uint32_t *message_length, uint8_t *message);
تعریفنوع صحیح (*fastrpc_mmap_pfn_t)(صحیح domain, صحیح fd, تهی *addr, صحیح offset, size_t length, شمارش fastrpc_map_flags flags);
تعریفنوع صحیح (*fastrpc_munmap_pfn_t)(صحیح domain, صحیح fd, تهی *addr, size_t length);

تعریفنوع صحیح (*remote_handle64_open_pfn_t)(ثابت نویسه* name, remote_handle64 *ph);
تعریفنوع صحیح (*remote_handle64_invoke_pfn_t)(remote_handle64 h, uint32_t dwScalars, remote_arg *pra);
تعریفنوع صحیح (*remote_handle64_close_pfn_t)(remote_handle h);
تعریفنوع صحیح (*remote_handle_control_pfn_t)(uint32_t req, تهی* data, uint32_t datalen);
تعریفنوع صحیح (*remote_handle64_control_pfn_t)(remote_handle64 h, uint32_t req, تهی* data, uint32_t datalen);
تعریفنوع صحیح (*remote_session_control_pfn_t)(uint32_t req, تهی *data, uint32_t datalen);
تعریفنوع صحیح (*remote_system_request_pfn_t)(system_req_payload * req);

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

تهی * rpcmem_alloc(صحیح heapid, uint32_t flags, صحیح size) {
    بازگشت rpcmem_alloc_pfn(heapid, flags, size);
}

تهی * rpcmem_alloc2(صحیح heapid, uint32_t flags, size_t size) {
    اگر (rpcmem_alloc2_pfn) {
        بازگشت rpcmem_alloc2_pfn(heapid, flags, size);
    } وگرنه {
        GGML_LOG_INFO("ggml-hex: rpcmem_alloc2 not found, falling back to rpcmem_alloc\n");
        بازگشت rpcmem_alloc_pfn(heapid, flags, size);
    }
}

تهی rpcmem_free(تهی * po) {
    بازگشت rpcmem_free_pfn(po);
}

صحیح rpcmem_to_fd(تهی * po) {
    بازگشت rpcmem_to_fd_pfn(po);
}

HTPDRV_API صحیح fastrpc_mmap(صحیح domain, صحیح fd, تهی * addr, صحیح offset, size_t length, شمارش fastrpc_map_flags flags) {
    بازگشت fastrpc_mmap_pfn(domain, fd, addr, offset, length, flags);
}

HTPDRV_API صحیح fastrpc_munmap(صحیح domain, صحیح fd, تهی * addr, size_t length) {
    بازگشت fastrpc_munmap_pfn(domain, fd, addr, length);
}

AEEResult dspqueue_create(صحیح                 domain,
                          uint32_t            flags,
                          uint32_t            req_queue_size,
                          uint32_t            resp_queue_size,
                          dspqueue_callback_t packet_callback,
                          dspqueue_callback_t error_callback,
                          تهی *              callback_context,
                          dspqueue_t *        queue) {
    بازگشت dspqueue_create_pfn(domain, flags, req_queue_size, resp_queue_size, packet_callback, error_callback,
                               callback_context, queue);
}

AEEResult dspqueue_close(dspqueue_t queue) {
    بازگشت dspqueue_close_pfn(queue);
}

AEEResult dspqueue_export(dspqueue_t queue, uint64_t * queue_id) {
    بازگشت dspqueue_export_pfn(queue, queue_id);
}

AEEResult dspqueue_write(dspqueue_t               queue,
                         uint32_t                 flags,
                         uint32_t                 رقم_buffers,
                         ساختار dspqueue_buffer * buffers,
                         uint32_t                 message_length,
                         ثابت uint8_t *          message,
                         uint32_t                 timeout_us) {
    بازگشت dspqueue_write_pfn(queue, flags, رقم_buffers, buffers, message_length, message, timeout_us);
}

AEEResult dspqueue_read(dspqueue_t               queue,
                        uint32_t *               flags,
                        uint32_t                 max_buffers,
                        uint32_t *               رقم_buffers,
                        ساختار dspqueue_buffer * buffers,
                        uint32_t                 max_message_length,
                        uint32_t *               message_length,
                        uint8_t *                message,
                        uint32_t                 timeout_us) {
#ifdef _WIN32
    اگر (timeout_us == 0) {
        بازگشت dspqueue_read_noblock_pfn(queue, flags, max_buffers, رقم_buffers, buffers, max_message_length,
                                     message_length, message);
    }
#endif
    بازگشت dspqueue_read_pfn(queue, flags, max_buffers, رقم_buffers, buffers, max_message_length, message_length,
                             message, timeout_us);
}

HTPDRV_API صحیح remote_handle64_open(ثابت نویسه * name, remote_handle64 * ph) {
    بازگشت remote_handle64_open_pfn(name, ph);
}

HTPDRV_API صحیح remote_handle64_invoke(remote_handle64 h, uint32_t dwScalars, remote_arg * pra) {
    بازگشت remote_handle64_invoke_pfn(h, dwScalars, pra);
}

HTPDRV_API صحیح remote_handle64_close(remote_handle64 h) {
    بازگشت remote_handle64_close_pfn(h);
}

HTPDRV_API صحیح remote_handle_control(uint32_t req, تهی * data, uint32_t datalen) {
    بازگشت remote_handle_control_pfn(req, data, datalen);
}

HTPDRV_API صحیح remote_handle64_control(remote_handle64 h, uint32_t req, تهی * data, uint32_t datalen) {
    بازگشت remote_handle64_control_pfn(h, req, data, datalen);
}

HTPDRV_API صحیح remote_session_control(uint32_t req, تهی * data, uint32_t datalen) {
    بازگشت remote_session_control_pfn(req, data, datalen);
}

HTPDRV_API صحیح remote_system_request(system_req_payload * req) {
    اگر (!remote_system_request_pfn) {
        بازگشت AEE_EUNSUPPORTEDAPI;
    }
    بازگشت remote_system_request_pfn(req);
}

#ifdef _WIN32

ایستا std::string wstr_to_str(std::wstring_view wstr) {
    std::string result;
    اگر (wstr.empty()) {
        بازگشت result;
    }
    auto bytes_needed = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.data(), (صحیح) wstr.size(),
                                            nullptr, 0, nullptr, nullptr);
    اگر (bytes_needed == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        پرتابکن std::runtime_error("Invalid wstring input");
    }

    result.resize(bytes_needed, '\0');
    صحیح bytes_written = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.data(), (صحیح) wstr.size(),
                                            result.data(), bytes_needed,
                                            nullptr, nullptr);
    اگر (bytes_written == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        پرتابکن std::runtime_error("Wstring conversion failed");
    }
    بازگشت result;
}

ایستا std::string get_driver_path() {
    std::wstring serviceName = L"qcnspmcdm";
    std::string result;

    // Get a handle to the SCM database.
    SC_HANDLE schSCManager = OpenSCManagerW(پوچ, پوچ, STANDARD_RIGHTS_READ);
    اگر (nullptr == schSCManager) {
        GGML_LOG_ERROR("ggml-hex: Failed to open SCManager. Error: %lu\n", GetLastError());
        بازگشت result;
    }

    // Get a handle to the service.
    SC_HANDLE schService = OpenServiceW(schSCManager,           // SCM database
                                        serviceName.c_str(),    // name of service
                                        SERVICE_QUERY_CONFIG);  // need query config access

    اگر (nullptr == schService) {
        GGML_LOG_ERROR("ggml-hex: Failed to open qcnspmcdm service. Error: %lu\n", GetLastError());
        CloseServiceHandle(schSCManager);
        بازگشت result;
    }

    // Store the size of buffer used as an output.
    DWORD bufferSize;
    اگر (!QueryServiceConfigW(schService, پوچ, 0, &bufferSize) &&
        (GetLastError() != ERROR_INSUFFICIENT_BUFFER)) {
        GGML_LOG_ERROR("ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        بازگشت result;
    }
    // Get the configuration of the service.
    LPQUERY_SERVICE_CONFIGW serviceConfig =
        static_cast<LPQUERY_SERVICE_CONFIGW>(LocalAlloc(LMEM_FIXED, bufferSize));
    اگر (!QueryServiceConfigW(schService, serviceConfig, bufferSize, &bufferSize)) {
        fprintf(stderr, "ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        LocalFree(serviceConfig);
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        بازگشت result;
    }

    // Read the driver file path get its parent directory
    std::wstring driverPath = std::wstring(serviceConfig->lp_دودویی_Path_Name);
    driverPath = driverPath.substr(0, driverPath.find_last_of(L"\\"));

    // Clean up resources
    LocalFree(serviceConfig);
    CloseServiceHandle(schService);
    CloseServiceHandle(schSCManager);

    // Driver path would contain invalid path string, like:
    // \SystemRoot\System32\DriverStore\FileRepository\qcadsprpc8280.inf_arm64_c2b9460c9a072f37
    // "\SystemRoot" should be replace with a correct one (e.g. C:\Windows)
    ثابت std::wstring system_ریشه_Placeholder = L"\\SystemRoot";
    اگر (0 != driverPath.compare(0, system_ریشه_Placeholder.length(), system_ریشه_Placeholder)) {
        GGML_LOG_ERROR("ggml-hex: String pattern not found in driver path.\n");
        بازگشت result;
    }

    // Replace \SystemRoot with an absolute path from system ENV windir
    ثابت std::wstring system_ریشه_Env = L"windir";

    // Query the number of wide characters this variable requires
    DWORD رقم_واژگان = GetEnvironmentVariableW(system_ریشه_Env.c_str(), پوچ, 0);
    اگر (رقم_واژگان == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed get systemRoot environment variable\n");
        بازگشت result;
    }

    // Query the actual system root name from environment variable
    std::vector<wchar_t> system_ریشه(رقم_واژگان + 1);
    رقم_واژگان = GetEnvironmentVariableW(system_ریشه_Env.c_str(), system_ریشه.data(), رقم_واژگان + 1);
    اگر (رقم_واژگان == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed to read windir environment variable\n");
        بازگشت result;
    }
    driverPath.replace(0, system_ریشه_Placeholder.length(), std::wstring(system_ریشه.data()));

    بازگشت wstr_to_str(driverPath);
}

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

صحیح htpdrv_init() {
    ایستا dl_handle_ptr lib_cdsp_rpc_handle = nullptr;
    ایستا منطقی initialized = نادرست;
#ifdef _WIN32
    std::string drv_path = get_driver_path() + "\\" + "libcdsprpc.dll";
#وگرنه
    std::string drv_path = "libcdsprpc.so";
#endif
    اگر (initialized) {
        GGML_LOG_INFO("ggml-hex: Driver already loaded\n");
        بازگشت AEE_SUCCESS;
    }
    GGML_LOG_INFO("ggml-hex: Loading driver %s\n", drv_path.c_str());

    fs::path path{ drv_path.c_str() };
    dl_handle_ptr handle { dl_load_library(path) };
    اگر (!handle) {
        GGML_LOG_ERROR("ggml-hex: failed to load %s: %s\n", path.u8string().c_str(), dl_error());
        بازگشت AEE_EUNABLETOLOAD;
    }

#define dlsym(drv, type, pfn, نماد, ignore)                               \
    انجامبده {                                                                    \
        pfn = (type) dl_get_sym(drv, #symbol);                              \
        اگر (!ignore && nullptr == pfn) {                                    \
            GGML_LOG_ERROR("ggml-hex: failed to dlsym %s\n", #symbol);      \
            بازگشت AEE_EUNABLETOLOAD;                                       \
        }                                                                   \
    } وقتی (0)

    dlsym(handle.get(), rpcmem_alloc_pfn_t, rpcmem_alloc_pfn, rpcmem_alloc, نادرست);
    dlsym(handle.get(), rpcmem_alloc2_pfn_t, rpcmem_alloc2_pfn, rpcmem_alloc2, درست);
    dlsym(handle.get(), rpcmem_free_pfn_t, rpcmem_free_pfn, rpcmem_free, نادرست);
    dlsym(handle.get(), rpcmem_to_fd_pfn_t, rpcmem_to_fd_pfn, rpcmem_to_fd, نادرست);
    dlsym(handle.get(), fastrpc_mmap_pfn_t, fastrpc_mmap_pfn, fastrpc_mmap, نادرست);
    dlsym(handle.get(), fastrpc_munmap_pfn_t, fastrpc_munmap_pfn, fastrpc_munmap, نادرست);
    dlsym(handle.get(), dspqueue_create_pfn_t, dspqueue_create_pfn, dspqueue_create, نادرست);
    dlsym(handle.get(), dspqueue_close_pfn_t, dspqueue_close_pfn, dspqueue_close, نادرست);
    dlsym(handle.get(), dspqueue_export_pfn_t, dspqueue_export_pfn, dspqueue_export, نادرست);
    dlsym(handle.get(), dspqueue_write_pfn_t, dspqueue_write_pfn, dspqueue_write, نادرست);
    dlsym(handle.get(), dspqueue_read_pfn_t, dspqueue_read_pfn, dspqueue_read, نادرست);
    dlsym(handle.get(), dspqueue_read_noblock_pfn_t, dspqueue_read_noblock_pfn, dspqueue_read_noblock, نادرست);
    dlsym(handle.get(), remote_handle64_open_pfn_t, remote_handle64_open_pfn, remote_handle64_open, نادرست);
    dlsym(handle.get(), remote_handle64_invoke_pfn_t, remote_handle64_invoke_pfn, remote_handle64_invoke, نادرست);
    dlsym(handle.get(), remote_handle_control_pfn_t, remote_handle_control_pfn, remote_handle_control, نادرست);
    dlsym(handle.get(), remote_handle64_control_pfn_t, remote_handle64_control_pfn, remote_handle64_control, نادرست);
    dlsym(handle.get(), remote_session_control_pfn_t, remote_session_control_pfn, remote_session_control, نادرست);
    dlsym(handle.get(), remote_handle64_close_pfn_t, remote_handle64_close_pfn, remote_handle64_close, نادرست);
    dlsym(handle.get(), remote_system_request_pfn_t, remote_system_request_pfn, remote_system_request, درست);

    lib_cdsp_rpc_handle = std::move(handle);
    initialized         = درست;

    بازگشت AEE_SUCCESS;
}

domain * htpdrv_get_domain(صحیح domain_id) {
    صحیح i    = 0;
    صحیح size = اندازه(supported_domains) / اندازه(domain);

    برای (i = 0; i < size; i++) {
        اگر (supported_domains[i].id == domain_id) {
            بازگشت &supported_domains[i];
        }
    }

    بازگشت پوچ;
}

صحیح htpdrv_get_arch(صحیح domain, صحیح * arch) {
    اگر (!remote_handle_control_pfn) {
        GGML_LOG_ERROR("ggml-hex: remote_handle_control is not supported on this device\n");
        بازگشت AEE_EUNSUPPORTEDAPI;
    }

    ساختار remote_dsp_capability arch_ver;
    arch_ver.domain       = (uint32_t) domain;
    arch_ver.attribute_ID = ARCH_VER;
    arch_ver.capability   = (uint32_t) 0;

    صحیح err = remote_handle_control(DSPRPC_GET_DSP_INFO, &arch_ver, اندازه(arch_ver));
    اگر ((err & 0xff) == (AEE_EUNSUPPORTEDAPI & 0xff)) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability API is not supported on this device\n");
        بازگشت AEE_EUNSUPPORTEDAPI;
    }

    اگر (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability query failed (err %d)\n", err);
        بازگشت err;
    }

    uint32_t val = arch_ver.capability & 0xff;
    *arch = (صحیح) ((val >> 4) * 10 + (val & 0x0f));
    بازگشت 0;
}
