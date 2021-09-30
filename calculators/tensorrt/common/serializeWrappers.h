#ifndef TENSORRT_SERIALIZE_WRAPPERS_H
#define TENSORRT_SERIALIZE_WRAPPERS_H

//!
//! \file serializeWrappers.h
//! Header which implements wrapper templates around TensorRT serialization methods that print CRC32 checksums of the
//! serialized engine for debugging purposes.  Templates are necessary here since the actual type of IRuntime, IEngine,
//! etc. depends on whether the sample has been compiled to work with the standard runtime, safety runtime, or builder.
//!
//! \warning Note that CRC32 is *not* suitable as a cryptograpic hash algorithm and thus should not be used in
//! production code to check for copying errors or malicious modifications of the serialized engine (see TRT_CC_RES_02
//! in the TensorRT Consistency Checker Software Tool Safety Manual).  These wrapper templates may be removed in a
//! future version of the TensorRT sample code.

#include "crc32.h"
#include "logger.h"

namespace samplesCommon
{

namespace
{

// Wrapper around ICudaEngine::serialize which prints the CRC32 checksum and size in bytes of the serialized engine
template <typename TCudaEngine>
inline auto serializeEngine(TCudaEngine* engine) -> decltype(engine->serialize())
{
    auto ret = engine->serialize();
    ASSERT(ret != nullptr);
    sample::gLogInfo << "Serialized engine, crc32=" << sample::crc::crc32(ret->data(), ret->size())
                     << ", size=" << ret->size() << std::endl;
    return ret;
}

// Wrapper around IBuilder::buildSerializedNetwork which prints the CRC32 checksum
// and size in bytes of the serialized engine.
template <typename TBuilder, typename TNetworkDefinition, typename TBuilderConfig>
inline auto buildSerializedNetwork(TBuilder* builder, TNetworkDefinition& network, TBuilderConfig& cfg)
    -> decltype(builder->buildSerializedNetwork(network, cfg))
{
    auto ret = builder->buildSerializedNetwork(network, cfg);
    if (ret != nullptr)
    {
        sample::gLogInfo << "Built serialized network, crc32=" << sample::crc::crc32(ret->data(), ret->size())
                         << ", size=" << ret->size() << std::endl;
    }
    return ret;
}

// Wrapper around IRuntime::deserialize which prints the CRC32 checksum and size in bytes of the serialized engine
template <typename TRuntimePtr, typename TPluginFactoryPtr>
inline auto deserializeCudaEngine(TRuntimePtr runtime, const void* data, size_t size, TPluginFactoryPtr pluginFactory)
    -> decltype(runtime->deserializeCudaEngine(data, size))
{
    sample::gLogInfo << "Deserializing engine, crc32=" << sample::crc::crc32(data, size) << ", size=" << size
                     << std::endl;
    auto ret = runtime->deserializeCudaEngine(data, size, pluginFactory);
    ASSERT(ret != nullptr);
    return ret;
}

template <typename TRuntimePtr>
inline auto deserializeCudaEngine(TRuntimePtr runtime, const void* data, size_t size, std::nullptr_t = nullptr)
    -> decltype(runtime->deserializeCudaEngine(data, size))
{
    sample::gLogInfo << "Deserializing engine, crc32=" << sample::crc::crc32(data, size) << ", size=" << size
                     << std::endl;
    auto ret = runtime->deserializeCudaEngine(data, size);
    ASSERT(ret != nullptr);
    return ret;
}

} // namespace
} // namespace samplesCommon

#endif // TENSORRT_SERIALIZE_WRAPPERS_H
