// Native timing for the installed CANN ADXL API. Build against that installation's headers.
#include <adxl/adxl_engine.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

namespace {

uint64_t timestamp(clockid_t clock) {
    timespec value{};
    clock_gettime(clock, &value);
    return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL + value.tv_nsec;
}

template <typename Function>
Function resolve(const char* name) {
    void* symbol = dlsym(RTLD_NEXT, name);
    if (!symbol) {
        // Python extension dependencies may be in an RTLD_LOCAL group.
        void* handle = dlopen("libllm_datadist.so", RTLD_NOW | RTLD_NOLOAD);
        if (handle) {
            symbol = dlsym(handle, name);
        }
    }
    if (!symbol) {
        std::fputs("ADXL profiling could not resolve the native symbol\n", stderr);
        std::abort();
    }
    return reinterpret_cast<Function>(symbol);
}

void emit(const char* kind, int op, size_t descriptors, uint64_t bytes, uint64_t begin, uint64_t end,
          uint32_t status) {
    char buffer[480];
    int count = std::snprintf(
        buffer, sizeof(buffer),
        "ISSUE23_OBSERVATION {\"type\":\"%s\",\"pid\":%ld,\"thread_id\":%ld,\"unix_ns\":%llu,"
        "\"monotonic_ns\":%llu,\"begin_ns\":%llu,\"end_ns\":%llu,\"operation\":%d,"
        "\"descriptor_count\":%zu,\"payload_bytes\":%llu,\"status\":%u}\n",
        kind, static_cast<long>(getpid()), syscall(SYS_gettid), static_cast<unsigned long long>(timestamp(CLOCK_REALTIME)),
        static_cast<unsigned long long>(end), static_cast<unsigned long long>(begin),
        static_cast<unsigned long long>(end), op, descriptors, static_cast<unsigned long long>(bytes), status);
    if (count > 0 && static_cast<size_t>(count) < sizeof(buffer)) {
        (void)write(STDERR_FILENO, buffer, static_cast<size_t>(count));
    }
}

}  // namespace

adxl::Status adxl::AdxlEngine::TransferSync(const AscendString& peer, TransferOp op,
                                            const std::vector<TransferOpDesc>& descriptors, int32_t timeout) {
    using Native = Status (*)(AdxlEngine*, const AscendString&, TransferOp,
                              const std::vector<TransferOpDesc>&, int32_t);
    static const auto native = resolve<Native>(
        "_ZN4adxl10AdxlEngine12TransferSyncERKN2ge12AscendStringENS_10TransferOpERKSt6vectorINS_14TransferOpDescESaIS7_EEi");
    const uint64_t begin = timestamp(CLOCK_MONOTONIC);
    const Status result = native(this, peer, op, descriptors, timeout);
    const uint64_t end = timestamp(CLOCK_MONOTONIC);
    uint64_t bytes = 0;
    for (const auto& row : descriptors) {
        bytes += row.len;
    }
    emit("adxl_transfer_sync", op, descriptors.size(), bytes, begin, end, result);
    return result;
}

adxl::Status adxl::AdxlEngine::Connect(const AscendString& peer, int32_t timeout) {
    using Native = Status (*)(AdxlEngine*, const AscendString&, int32_t);
    static const auto native = resolve<Native>("_ZN4adxl10AdxlEngine7ConnectERKN2ge12AscendStringEi");
    const uint64_t begin = timestamp(CLOCK_MONOTONIC);
    const Status result = native(this, peer, timeout);
    emit("adxl_connect", -1, 0, 0, begin, timestamp(CLOCK_MONOTONIC), result);
    return result;
}
