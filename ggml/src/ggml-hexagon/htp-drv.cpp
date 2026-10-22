#include <filesystem>
#include <set>
#include <sstream>
#include <string>

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#否则
#   include <dlfcn.h>
#   include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"
#pragma clang diagnostic ignored "-Wnested-anon-types"

#include "ggml-impl.h"
#include "htp-drv.h"
#include "libdl.h"

#include <domain.h>

//
// Driver API types
//

typedef void * (*rpcmem_alloc_pfn_t)(整数 heapid, uint32_t flags, 整数 size);
typedef void * (*rpcmem_alloc2_pfn_t)(整数 heapid, uint32_t flags, size_t size);
typedef void   (*rpcmem_free_pfn_t)(void * po);
typedef 整数    (*rpcmem_to_fd_pfn_t)(void * po);

typedef AEEResult (*dspqueue_create_pfn_t)(整数                 domain,
                                           uint32_t            flags,
                                           uint32_t            req_queue_size,
                                           uint32_t            resp_queue_size,
                                           dspqueue_callback_t packet_callback,
                                           dspqueue_callback_t error_callback,
                                           void *              callback_context,
                                           dspqueue_t *        queue);
typedef AEEResult (*dspqueue_close_pfn_t)(dspqueue_t queue);
typedef AEEResult (*dspqueue_export_pfn_t)(dspqueue_t queue, uint64_t *queue_id);
typedef AEEResult (*dspqueue_write_pfn_t)(dspqueue_t queue, uint32_t flags,
                                          uint32_t num_buffers,
                                          struct dspqueue_buffer *buffers,
                                          uint32_t message_length,
                                          const uint8_t *message,
                                          uint32_t timeout_us);
typedef AEEResult (*dspqueue_read_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                         uint32_t max_buffers, uint32_t *num_buffers,
                                         struct dspqueue_buffer *buffers,
                                         uint32_t max_message_length,
                                         uint32_t *message_length, uint8_t *message,
                                         uint32_t timeout_us);
typedef AEEResult (*dspqueue_read_noblock_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                                 uint32_t max_buffers, uint32_t *num_buffers,
                                                 struct dspqueue_buffer *buffers,
                                                 uint32_t max_message_length,
                                                 uint32_t *message_length, uint8_t *message);
typedef 整数 (*fastrpc_mmap_pfn_t)(整数 domain, 整数 fd, void *addr, 整数 offset, size_t length, enum fastrpc_map_flags flags);
typedef 整数 (*fastrpc_munmap_pfn_t)(整数 domain, 整数 fd, void *addr, size_t length);

typedef 整数 (*remote_handle64_open_pfn_t)(const char* name, remote_handle64 *ph);
typedef 整数 (*remote_handle64_invoke_pfn_t)(remote_handle64 h, uint32_t dwScalars, remote_arg *pra);
typedef 整数 (*remote_handle64_close_pfn_t)(remote_handle h);
typedef 整数 (*remote_handle_control_pfn_t)(uint32_t req, void* data, uint32_t datalen);
typedef 整数 (*remote_handle64_control_pfn_t)(remote_handle64 h, uint32_t req, void* data, uint32_t datalen);
typedef 整数 (*remote_session_control_pfn_t)(uint32_t req, void *data, uint32_t datalen);
typedef 整数 (*remote_system_request_pfn_t)(system_req_payload * req);

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

void * rpcmem_alloc(整数 heapid, uint32_t flags, 整数 size) {
    返回 rpcmem_alloc_pfn(heapid, flags, size);
}

void * rpcmem_alloc2(整数 heapid, uint32_t flags, size_t size) {
    如果 (rpcmem_alloc2_pfn) {
        返回 rpcmem_alloc2_pfn(heapid, flags, size);
    } 否则 {
        GGML_LOG_INFO("ggml-hex: rpcmem_alloc2 not found, falling back to rpcmem_alloc\n");
        返回 rpcmem_alloc_pfn(heapid, flags, size);
    }
}

void rpcmem_free(void * po) {
    返回 rpcmem_free_pfn(po);
}

整数 rpcmem_to_fd(void * po) {
    返回 rpcmem_to_fd_pfn(po);
}

HTPDRV_API 整数 fastrpc_mmap(整数 domain, 整数 fd, void * addr, 整数 offset, size_t length, enum fastrpc_map_flags flags) {
    返回 fastrpc_mmap_pfn(domain, fd, addr, offset, length, flags);
}

HTPDRV_API 整数 fastrpc_munmap(整数 domain, 整数 fd, void * addr, size_t length) {
    返回 fastrpc_munmap_pfn(domain, fd, addr, length);
}

AEEResult dspqueue_create(整数                 domain,
                          uint32_t            flags,
                          uint32_t            req_queue_size,
                          uint32_t            resp_queue_size,
                          dspqueue_callback_t packet_callback,
                          dspqueue_callback_t error_callback,
                          void *              callback_context,
                          dspqueue_t *        queue) {
    返回 dspqueue_create_pfn(domain, flags, req_queue_size, resp_queue_size, packet_callback, error_callback,
                               callback_context, queue);
}

AEEResult dspqueue_close(dspqueue_t queue) {
    返回 dspqueue_close_pfn(queue);
}

AEEResult dspqueue_export(dspqueue_t queue, uint64_t * queue_id) {
    返回 dspqueue_export_pfn(queue, queue_id);
}

AEEResult dspqueue_write(dspqueue_t               queue,
                         uint32_t                 flags,
                         uint32_t                 num_buffers,
                         struct dspqueue_buffer * buffers,
                         uint32_t                 message_length,
                         const uint8_t *          message,
                         uint32_t                 timeout_us) {
    返回 dspqueue_write_pfn(queue, flags, num_buffers, buffers, message_length, message, timeout_us);
}

AEEResult dspqueue_read(dspqueue_t               queue,
                        uint32_t *               flags,
                        uint32_t                 max_buffers,
                        uint32_t *               num_buffers,
                        struct dspqueue_buffer * buffers,
                        uint32_t                 max_message_length,
                        uint32_t *               message_length,
                        uint8_t *                message,
                        uint32_t                 timeout_us) {
#ifdef _WIN32
    如果 (timeout_us == 0) {
        返回 dspqueue_read_noblock_pfn(queue, flags, max_buffers, num_buffers, buffers, max_message_length,
                                     message_length, message);
    }
#endif
    返回 dspqueue_read_pfn(queue, flags, max_buffers, num_buffers, buffers, max_message_length, message_length,
                             message, timeout_us);
}

HTPDRV_API 整数 remote_handle64_open(const char * name, remote_handle64 * ph) {
    返回 remote_handle64_open_pfn(name, ph);
}

HTPDRV_API 整数 remote_handle64_invoke(remote_handle64 h, uint32_t dwScalars, remote_arg * pra) {
    返回 remote_handle64_invoke_pfn(h, dwScalars, pra);
}

HTPDRV_API 整数 remote_handle64_close(remote_handle64 h) {
    返回 remote_handle64_close_pfn(h);
}

HTPDRV_API 整数 remote_handle_control(uint32_t req, void * data, uint32_t datalen) {
    返回 remote_handle_control_pfn(req, data, datalen);
}

HTPDRV_API 整数 remote_handle64_control(remote_handle64 h, uint32_t req, void * data, uint32_t datalen) {
    返回 remote_handle64_control_pfn(h, req, data, datalen);
}

HTPDRV_API 整数 remote_session_control(uint32_t req, void * data, uint32_t datalen) {
    返回 remote_session_control_pfn(req, data, datalen);
}

HTPDRV_API 整数 remote_system_request(system_req_payload * req) {
    如果 (!remote_system_request_pfn) {
        返回 AEE_EUNSUPPORTEDAPI;
    }
    返回 remote_system_request_pfn(req);
}

#ifdef _WIN32

static std::string wstr_to_str(std::wstring_view wstr) {
    std::string result;
    如果 (wstr.empty()) {
        返回 result;
    }
    auto bytes_needed = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.data(), (整数) wstr.size(),
                                            nullptr, 0, nullptr, nullptr);
    如果 (bytes_needed == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        throw std::runtime_error("Invalid wstring input");
    }

    result.resize(bytes_needed, '\0');
    整数 bytes_written = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.data(), (整数) wstr.size(),
                                            result.data(), bytes_needed,
                                            nullptr, nullptr);
    如果 (bytes_written == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        throw std::runtime_error("Wstring conversion failed");
    }
    返回 result;
}

static std::string get_driver_path() {
    std::wstring serviceName = L"qcnspmcdm";
    std::string result;

    // Get a handle to the SCM database.
    SC_HANDLE schSCManager = OpenSCManagerW(NULL, NULL, STANDARD_RIGHTS_READ);
    如果 (nullptr == schSCManager) {
        GGML_LOG_ERROR("ggml-hex: Failed to open SCManager. Error: %lu\n", GetLastError());
        返回 result;
    }

    // Get a handle to the service.
    SC_HANDLE schService = OpenServiceW(schSCManager,           // SCM database
                                        serviceName.c_str(),    // name of service
                                        SERVICE_QUERY_CONFIG);  // need query config access

    如果 (nullptr == schService) {
        GGML_LOG_ERROR("ggml-hex: Failed to open qcnspmcdm service. Error: %lu\n", GetLastError());
        CloseServiceHandle(schSCManager);
        返回 result;
    }

    // Store the size of buffer used as an output.
    DWORD bufferSize;
    如果 (!QueryServiceConfigW(schService, NULL, 0, &bufferSize) &&
        (GetLastError() != ERROR_INSUFFICIENT_BUFFER)) {
        GGML_LOG_ERROR("ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        返回 result;
    }
    // Get the configuration of the service.
    LPQUERY_SERVICE_CONFIGW serviceConfig =
        static_cast<LPQUERY_SERVICE_CONFIGW>(LocalAlloc(LMEM_FIXED, bufferSize));
    如果 (!QueryServiceConfigW(schService, serviceConfig, bufferSize, &bufferSize)) {
        fprintf(stderr, "ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        LocalFree(serviceConfig);
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        返回 result;
    }

    // Read the driver file path get its parent directory
    std::wstring driverPath = std::wstring(serviceConfig->lpBinaryPathName);
    driverPath = driverPath.substr(0, driverPath.find_last_of(L"\\"));

    // Clean up resources
    LocalFree(serviceConfig);
    CloseServiceHandle(schService);
    CloseServiceHandle(schSCManager);

    // Driver path would contain invalid path string, like:
    // \SystemRoot\System32\DriverStore\FileRepository\qcadsprpc8280.inf_arm64_c2b9460c9a072f37
    // "\SystemRoot" should be replace with a correct one (e.g. C:\Windows)
    const std::wstring systemRootPlaceholder = L"\\SystemRoot";
    如果 (0 != driverPath.compare(0, systemRootPlaceholder.length(), systemRootPlaceholder)) {
        GGML_LOG_ERROR("ggml-hex: String pattern not found in driver path.\n");
        返回 result;
    }

    // Replace \SystemRoot with an absolute path from system ENV windir
    const std::wstring systemRootEnv = L"windir";

    // Query the number of wide characters this variable requires
    DWORD numWords = GetEnvironmentVariableW(systemRootEnv.c_str(), NULL, 0);
    如果 (numWords == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed get systemRoot environment variable\n");
        返回 result;
    }

    // Query the actual system root name from environment variable
    std::vector<wchar_t> systemRoot(numWords + 1);
    numWords = GetEnvironmentVariableW(systemRootEnv.c_str(), systemRoot.data(), numWords + 1);
    如果 (numWords == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed to read windir environment variable\n");
        返回 result;
    }
    driverPath.replace(0, systemRootPlaceholder.length(), std::wstring(systemRoot.data()));

    返回 wstr_to_str(driverPath);
}

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

整数 htpdrv_init() {
    static dl_handle_ptr lib_cdsp_rpc_handle = nullptr;
    static bool initialized = false;
#ifdef _WIN32
    std::string drv_path = get_driver_path() + "\\" + "libcdsprpc.dll";
#否则
    std::string drv_path = "libcdsprpc.so";
#endif
    如果 (initialized) {
        GGML_LOG_INFO("ggml-hex: Driver already loaded\n");
        返回 AEE_SUCCESS;
    }
    GGML_LOG_INFO("ggml-hex: Loading driver %s\n", drv_path.c_str());

    fs::path path{ drv_path.c_str() };
    dl_handle_ptr handle { dl_load_library(path) };
    如果 (!handle) {
        GGML_LOG_ERROR("ggml-hex: failed to load %s: %s\n", path.u8string().c_str(), dl_error());
        返回 AEE_EUNABLETOLOAD;
    }

#define dlsym(drv, type, pfn, symbol, ignore)                               \
    do {                                                                    \
        pfn = (type) dl_get_sym(drv, #symbol);                              \
        如果 (!ignore && nullptr == pfn) {                                    \
            GGML_LOG_ERROR("ggml-hex: failed to dlsym %s\n", #symbol);      \
            返回 AEE_EUNABLETOLOAD;                                       \
        }                                                                   \
    } 当 (0)

    dlsym(handle.get(), rpcmem_alloc_pfn_t, rpcmem_alloc_pfn, rpcmem_alloc, false);
    dlsym(handle.get(), rpcmem_alloc2_pfn_t, rpcmem_alloc2_pfn, rpcmem_alloc2, true);
    dlsym(handle.get(), rpcmem_free_pfn_t, rpcmem_free_pfn, rpcmem_free, false);
    dlsym(handle.get(), rpcmem_to_fd_pfn_t, rpcmem_to_fd_pfn, rpcmem_to_fd, false);
    dlsym(handle.get(), fastrpc_mmap_pfn_t, fastrpc_mmap_pfn, fastrpc_mmap, false);
    dlsym(handle.get(), fastrpc_munmap_pfn_t, fastrpc_munmap_pfn, fastrpc_munmap, false);
    dlsym(handle.get(), dspqueue_create_pfn_t, dspqueue_create_pfn, dspqueue_create, false);
    dlsym(handle.get(), dspqueue_close_pfn_t, dspqueue_close_pfn, dspqueue_close, false);
    dlsym(handle.get(), dspqueue_export_pfn_t, dspqueue_export_pfn, dspqueue_export, false);
    dlsym(handle.get(), dspqueue_write_pfn_t, dspqueue_write_pfn, dspqueue_write, false);
    dlsym(handle.get(), dspqueue_read_pfn_t, dspqueue_read_pfn, dspqueue_read, false);
    dlsym(handle.get(), dspqueue_read_noblock_pfn_t, dspqueue_read_noblock_pfn, dspqueue_read_noblock, false);
    dlsym(handle.get(), remote_handle64_open_pfn_t, remote_handle64_open_pfn, remote_handle64_open, false);
    dlsym(handle.get(), remote_handle64_invoke_pfn_t, remote_handle64_invoke_pfn, remote_handle64_invoke, false);
    dlsym(handle.get(), remote_handle_control_pfn_t, remote_handle_control_pfn, remote_handle_control, false);
    dlsym(handle.get(), remote_handle64_control_pfn_t, remote_handle64_control_pfn, remote_handle64_control, false);
    dlsym(handle.get(), remote_session_control_pfn_t, remote_session_control_pfn, remote_session_control, false);
    dlsym(handle.get(), remote_handle64_close_pfn_t, remote_handle64_close_pfn, remote_handle64_close, false);
    dlsym(handle.get(), remote_system_request_pfn_t, remote_system_request_pfn, remote_system_request, true);

    lib_cdsp_rpc_handle = std::move(handle);
    initialized         = true;

    返回 AEE_SUCCESS;
}

domain * htpdrv_get_domain(整数 domain_id) {
    整数 i    = 0;
    整数 size = sizeof(supported_domains) / sizeof(domain);

    为 (i = 0; i < size; i++) {
        如果 (supported_domains[i].id == domain_id) {
            返回 &supported_domains[i];
        }
    }

    返回 NULL;
}

整数 htpdrv_get_arch(整数 domain, 整数 * arch) {
    如果 (!remote_handle_control_pfn) {
        GGML_LOG_ERROR("ggml-hex: remote_handle_control is not supported on this device\n");
        返回 AEE_EUNSUPPORTEDAPI;
    }

    struct remote_dsp_capability arch_ver;
    arch_ver.domain       = (uint32_t) domain;
    arch_ver.attribute_ID = ARCH_VER;
    arch_ver.capability   = (uint32_t) 0;

    整数 err = remote_handle_control(DSPRPC_GET_DSP_INFO, &arch_ver, sizeof(arch_ver));
    如果 ((err & 0xff) == (AEE_EUNSUPPORTEDAPI & 0xff)) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability API is not supported on this device\n");
        返回 AEE_EUNSUPPORTEDAPI;
    }

    如果 (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability query failed (err %d)\n", err);
        返回 err;
    }

    uint32_t val = arch_ver.capability & 0xff;
    *arch = (整数) ((val >> 4) * 10 + (val & 0x0f));
    返回 0;
}
