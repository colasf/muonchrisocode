// Direct3D 11 device and the few helpers the commands share.
#pragma once
#include <d3d11_1.h>
#include <dxgi1_6.h>
#include <windows.h>
#include <wrl/client.h>

#include <chrono>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

inline double now()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Gpu {
    ComPtr<ID3D11Device> dev;
    ComPtr<ID3D11DeviceContext> ctx;
    ComPtr<IDXGIFactory6> factory;
    ComPtr<ID3D11Query> query;
    std::wstring name;

    // The fastest adapter of the machine (a laptop gives its integrated one by default).
    bool create(bool debug, std::string& err)
    {
        ComPtr<IDXGIAdapter1> adapter;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
            factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter));
        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT | (debug ? D3D11_CREATE_DEVICE_DEBUG : 0);
        D3D_FEATURE_LEVEL want = D3D_FEATURE_LEVEL_11_0, got;
        HRESULT hr = D3D11CreateDevice(adapter.Get(), adapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
                                       &want, 1, D3D11_SDK_VERSION, &dev, &got, &ctx);
        if (FAILED(hr)) {
            char b[96];
            snprintf(b, sizeof b, "D3D11CreateDevice failed (0x%08lX)", (unsigned long)hr);
            err = b;
            return false;
        }
        if (adapter) {
            DXGI_ADAPTER_DESC1 d;
            adapter->GetDesc1(&d);
            name = d.Description;
        }
        D3D11_QUERY_DESC qd = { D3D11_QUERY_EVENT, 0 };
        dev->CreateQuery(&qd, &query);
        return true;
    }

    // Time the GPU spends on what is drawn between begin() and end(): milliseconds, read with gpuMs() after finish().
    ComPtr<ID3D11Query> qDisjoint, qT0, qT1;
    void timeBegin()
    {
        if (!qDisjoint) {
            D3D11_QUERY_DESC d = { D3D11_QUERY_TIMESTAMP_DISJOINT, 0 }, t = { D3D11_QUERY_TIMESTAMP, 0 };
            dev->CreateQuery(&d, &qDisjoint);
            dev->CreateQuery(&t, &qT0);
            dev->CreateQuery(&t, &qT1);
        }
        ctx->Begin(qDisjoint.Get());
        ctx->End(qT0.Get());
    }
    void timeEnd()
    {
        ctx->End(qT1.Get());
        ctx->End(qDisjoint.Get());
    }
    // The device is gone (driver reset, card removed): nothing more will ever be drawn with it.
    bool removed() const { return dev && dev->GetDeviceRemovedReason() != S_OK; }

    // Wait for the result of a query. False if it will never come (device removed) or takes over 5 s.
    bool wait(ID3D11Query* q, void* data, UINT size)
    {
        const double t0 = now();
        for (unsigned spin = 0;; spin++) {
            if (ctx->GetData(q, data, size, 0) == S_OK) return true;
            if ((spin & 1023) == 1023 && (removed() || now() - t0 > 5.0)) return false;
        }
    }
    double gpuMs()
    {
        UINT64 a = 0, b = 0;
        D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj = {};
        if (!wait(qDisjoint.Get(), &dj, sizeof dj) || !wait(qT0.Get(), &a, sizeof a) || !wait(qT1.Get(), &b, sizeof b)) return -1.0;
        return dj.Disjoint || !dj.Frequency ? -1.0 : (double)(b - a) * 1e3 / (double)dj.Frequency;
    }

    // Wait until the GPU has done everything it was asked.
    void finish()
    {
        ctx->End(query.Get());
        BOOL done = FALSE;
        while (wait(query.Get(), &done, sizeof done) && !done) {}
    }

    // Pixels of a texture (rows packed).
    bool readback(ID3D11Texture2D* tex, size_t pixelBytes, std::vector<uint8_t>& out)
    {
        D3D11_TEXTURE2D_DESC td;
        tex->GetDesc(&td);
        td.Usage = D3D11_USAGE_STAGING;
        td.BindFlags = 0;
        td.MiscFlags = 0;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Texture2D> st;
        if (FAILED(dev->CreateTexture2D(&td, nullptr, &st))) return false;
        ctx->CopyResource(st.Get(), tex);
        D3D11_MAPPED_SUBRESOURCE m;
        if (FAILED(ctx->Map(st.Get(), 0, D3D11_MAP_READ, 0, &m))) return false;
        size_t row = (size_t)td.Width * pixelBytes;
        out.resize(row * td.Height);
        for (UINT y = 0; y < td.Height; y++) memcpy(&out[y * row], (const uint8_t*)m.pData + (size_t)y * m.RowPitch, row);
        ctx->Unmap(st.Get(), 0);
        return true;
    }
};
