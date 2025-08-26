#include <jni.h>
#include <android/log.h>

#include <string>
#include <vector>
#include <algorithm>
#include <thread>
#include <cstdint>
#include <cstring>
#include <unistd.h>     // sysconf(_SC_NPROCESSORS_ONLN)

#include "whisper.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  "WhisperJNI", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "WhisperJNI", __VA_ARGS__)

// ---- helpers ----
static int cpu_threads_default() {
    int n = 0;

#ifdef _SC_NPROCESSORS_ONLN
    long v = sysconf(_SC_NPROCESSORS_ONLN);
    if (v > 0) n = static_cast<int>(v);
#endif
    if (n <= 0) {
        unsigned hc = std::thread::hardware_concurrency();
        if (hc > 0) n = static_cast<int>(hc);
    }
    if (n <= 0) n = 1;
    return std::max(1, n - 1); // lass 1 Kern fürs System frei
}

// ------------------------------------------------------------
// jlong init(String modelPath)
extern "C" JNIEXPORT jlong JNICALL
Java_com_example_whisper_sync_NativeWhisper_init(JNIEnv* env, jclass, jstring jpath) {
    const char* cpath = env->GetStringUTFChars(jpath, nullptr);
    LOGI("init: path=%s", cpath ? cpath : "(null)");

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;            // CPU auf Android
    // cparams.dont_print_meta = true;  // weniger native Logs

    whisper_context* ctx = whisper_init_from_file_with_params(cpath, cparams);
    env->ReleaseStringUTFChars(jpath, cpath);

    if (!ctx) {
        LOGE("init: FAILED (ctx == null)");
        return 0;
    }
    LOGI("init: ctx=%p OK", (void*)ctx);
    return reinterpret_cast<jlong>(ctx);
}

// ------------------------------------------------------------
// String fullTranscribe(long ctx, float[] pcm, int sampleRate)
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_whisper_sync_NativeWhisper_fullTranscribe(
        JNIEnv* env, jclass,
        jlong jctx, jfloatArray jpcm, jint sample_rate) {

    auto* ctx = reinterpret_cast<whisper_context*>(jctx);
    if (!ctx) {
        LOGE("fullTranscribe: ctx is null");
        return env->NewStringUTF("");
    }

    const jsize n = env->GetArrayLength(jpcm);
    if (n <= 0) {
        LOGE("fullTranscribe: empty PCM array");
        return env->NewStringUTF("");
    }

    std::vector<float> pcm(n);
    env->GetFloatArrayRegion(jpcm, 0, n, pcm.data());

    LOGI("fullTranscribe: samples=%d sr=%d", (int)pcm.size(), (int)sample_rate);

    // Greedy-Sampling ist auf Mobile oft am stabilsten/schnellsten
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    // Threads & Ausgabe
    params.n_threads        = cpu_threads_default();
    params.print_realtime   = false;
    params.print_progress   = true;
    params.print_timestamps = true;
    params.print_special    = false;

    // Sprache/Strategie
    params.translate        = false;            // nicht nach EN übersetzen
    params.detect_language  = false;            // schneller, wenn Sprache bekannt
    params.language         = "en";             // passe an: "de", "en", ...

    // Timing / Segmentierung
    params.no_context       = true;             // keine Historie
    params.no_timestamps    = false;            // Timestamps zulassen
    params.max_len          = 0;                // keine harte Begrenzung
    params.token_timestamps = false;            // keine Token-Timestamps
    params.split_on_word    = true;

    // Robustheit gegen "Stille" Fehlklassifikation:
    params.suppress_blank   = false;
    // entferntes Feld: params.suppress_non_speech_tokens  -> NICHT mehr setzen
    params.no_speech_thold  = 0.10f;            // Standard ~0.6; niedriger -> weniger „Stille“
    params.logprob_thold    = -2.0f;            // sehr liberal
    params.entropy_thold    = -1.0f;            // deaktiviert

    // Greedy-Feintuning (Sub-Struct existiert in aktuellen headers)
    // params.greedy.best_of = 1; // default ist i.d.R. 1

    LOGI("fullTranscribe: calling whisper_full ... (threads=%d)", params.n_threads);
    const int r = whisper_full(ctx, params, pcm.data(), (int)pcm.size());
    if (r != 0) {
        LOGE("fullTranscribe: whisper_full returned %d", r);
        return env->NewStringUTF("");
    }

    // Segmente einsammeln
    const int n_segments = whisper_full_n_segments(ctx);
    std::string out;
    out.reserve(256);
    for (int i = 0; i < n_segments; ++i) {
        const char* txt = whisper_full_get_segment_text(ctx, i);
        if (txt && *txt) out += txt;
    }

    LOGI("fullTranscribe: done, segments=%d, out_len=%d", n_segments, (int)out.size());
    return env->NewStringUTF(out.c_str());
}

// ------------------------------------------------------------
// void free(long ctx)
extern "C" JNIEXPORT void JNICALL
Java_com_example_whisper_sync_NativeWhisper_free(JNIEnv*, jclass, jlong jctx) {
    auto* ctx = reinterpret_cast<whisper_context*>(jctx);
    if (ctx) {
        whisper_free(ctx);
        LOGI("free: ctx freed");
    }
}
