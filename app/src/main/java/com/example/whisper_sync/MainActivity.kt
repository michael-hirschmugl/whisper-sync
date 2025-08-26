package com.example.whisper_sync

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.pointerInteropFilter
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.whisper_sync.ui.theme.WhispersyncTheme
import kotlinx.coroutines.*
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

private const val TAG = "Whisper"

// ---- JNI-Wrapper (Namen müssen zu whisper_jni.cpp passen) ----
object NativeWhisper {
    init {
        try {
            System.loadLibrary("whisper_jni")
            Log.i("Whisper", "System.loadLibrary(whisper_jni) OK")
        } catch (t: Throwable) {
            Log.e("Whisper", "System.loadLibrary(whisper_jni) FAILED", t)
        }
    }
    @JvmStatic external fun init(modelPath: String): Long
    @JvmStatic external fun fullTranscribe(ctx: Long, pcm: FloatArray, sampleRate: Int): String
    @JvmStatic external fun free(ctx: Long)
}

class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "Whisper"
    }

    // === Audio-Konstanten für WAV/whisper ===
    private val SAMPLE_RATE = 16_000
    private val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
    private val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT

    // === Aufnahme (AudioRecord) ===
    private var audioRecord: AudioRecord? = null
    private var recordingJob: Job? = null
    @Volatile private var isRecordingFlag: Boolean = false

    // Live-Amplitude für die Wellenform (0..32767)
    @Volatile private var lastAmplitude: Int = 0

    // === Abspielen ===
    private var player: MediaPlayer? = null

    // === Dateien ===
    private lateinit var outputWavFile: File

    // === Whisper Context ===
    private var whisperCtx: Long = 0L

    // Für schnellen Start: tiny-Modell (in assets ablegen!)
    // Pfad innerhalb der Assets:
    private val modelAssetPath = "models/ggml-tiny.bin"

    private val requestRecordAudio =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            Log.i(TAG, "RECORD_AUDIO permission result = $granted")
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // Zieldatei als WAV (für whisper.cpp)
        outputWavFile = File(cacheDir, "last_recording.wav")
        Log.i(TAG, "WAV output path = ${outputWavFile.absolutePath}")

        setContent {
            WhispersyncTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    RecorderScreen(
                        isMicGranted   = { isRecordAudioGranted() },
                        askForMic      = { askForRecordAudio() },
                        startRecording = { startPcmRecording() },
                        stopRecording  = { stopPcmRecording() },
                        startPlayback  = { startPlayback() },
                        stopPlayback   = { stopPlayback() },
                        hasRecording   = { outputWavFile.exists() && outputWavFile.length() > 44 },
                        getAmplitude   = { getRecorderAmplitude() },
                        onTranscribe   = { onTranscribeRequested() }
                    )
                }
            }
        }
    }

    // ===== Permission =====
    private fun isRecordAudioGranted(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED

    private fun askForRecordAudio() {
        Log.d(TAG, "Requesting RECORD_AUDIO permission")
        requestRecordAudio.launch(Manifest.permission.RECORD_AUDIO)
    }

    // ===== Aufnahme: WAV via AudioRecord =====
    private fun startPcmRecording() {
        if (outputWavFile.exists()) {
            val del = outputWavFile.delete()
            Log.d(TAG, "Deleted old wav = $del")
        }

        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        val bufferSize = maxOf(minBuf, 4096)
        Log.d(TAG, "AudioRecord bufferSize = $bufferSize (min=$minBuf)")

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT,
            bufferSize
        )

        val raf = RandomAccessFile(outputWavFile, "rw")
        writeWavHeaderPlaceholder(raf, SAMPLE_RATE, 1, 16)
        Log.i(TAG, "WAV header placeholder written")

        isRecordingFlag = true
        audioRecord?.startRecording()
        Log.i(TAG, "AudioRecord.startRecording()")

        recordingJob = lifecycleScope.launch(Dispatchers.IO) {
            val shortBuf = ShortArray(bufferSize / 2)
            var dataBytes: Long = 0

            try {
                while (isActive && isRecordingFlag) {
                    val n = audioRecord?.read(shortBuf, 0, shortBuf.size) ?: 0
                    if (n > 0) {
                        lastAmplitude = estimateAmplitude(shortBuf, n)

                        val bb = ByteBuffer.allocate(n * 2).order(ByteOrder.LITTLE_ENDIAN)
                        for (i in 0 until n) bb.putShort(shortBuf[i])
                        bb.flip()
                        raf.write(bb.array(), 0, n * 2)
                        dataBytes += (n * 2)
                    }
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Recording loop failed", t)
            } finally {
                try {
                    finalizeWavHeader(raf, dataBytes, SAMPLE_RATE, 1, 16)
                    Log.i(TAG, "WAV finalized: dataBytes=$dataBytes totalSize=${44 + dataBytes}")
                } catch (t: Throwable) {
                    Log.e(TAG, "finalizeWavHeader failed", t)
                }
                try { raf.close() } catch (_: Throwable) {}
            }
        }
    }

    private fun stopPcmRecording() {
        isRecordingFlag = false
        audioRecord?.let { ar ->
            runCatching { ar.stop() }.onFailure { Log.w(TAG, "AudioRecord.stop() failed", it) }
            ar.release()
        }
        audioRecord = null
        recordingJob?.cancel()
        recordingJob = null
        val size = outputWavFile.length()
        Log.i(TAG, "Recording stopped. WAV size = $size")
        lastAmplitude = 0
    }

    // ===== WAV-Utilities =====
    private fun estimateAmplitude(buf: ShortArray, n: Int): Int {
        if (n <= 0) return 0
        var sumSq = 0.0
        for (i in 0 until n) {
            val v = buf[i].toDouble()
            sumSq += v * v
        }
        val rms = sqrt(sumSq / n)
        return min(rms.toInt(), 32767)
    }

    private fun writeWavHeaderPlaceholder(raf: RandomAccessFile, sampleRate: Int, channels: Int, bitsPerSample: Int) {
        val byteRate = sampleRate * channels * bitsPerSample / 8
        val blockAlign = (channels * bitsPerSample / 8).toShort()

        raf.seek(0)
        // RIFF
        raf.writeBytes("RIFF")
        raf.writeIntLE(0)
        raf.writeBytes("WAVE")
        // fmt
        raf.writeBytes("fmt ")
        raf.writeIntLE(16)
        raf.writeShortLE(1) // PCM
        raf.writeShortLE(channels.toShort())
        raf.writeIntLE(sampleRate)
        raf.writeIntLE(byteRate)
        raf.writeShortLE(blockAlign)
        raf.writeShortLE(bitsPerSample.toShort())
        // data
        raf.writeBytes("data")
        raf.writeIntLE(0)
    }

    private fun finalizeWavHeader(raf: RandomAccessFile, dataBytes: Long, sampleRate: Int, channels: Int, bitsPerSample: Int) {
        val riffSize = 36L + dataBytes
        raf.seek(4);  raf.writeIntLE(riffSize.toInt())
        raf.seek(40); raf.writeIntLE(dataBytes.toInt())
        raf.seek(44 + dataBytes)
    }

    private fun RandomAccessFile.writeIntLE(v: Int) {
        this.write(byteArrayOf(
            (v and 0xFF).toByte(),
            ((v shr 8) and 0xFF).toByte(),
            ((v shr 16) and 0xFF).toByte(),
            ((v shr 24) and 0xFF).toByte()
        ))
    }
    private fun RandomAccessFile.writeShortLE(v: Short) {
        this.write(byteArrayOf(
            (v.toInt() and 0xFF).toByte(),
            ((v.toInt() shr 8) and 0xFF).toByte()
        ))
    }

    // ===== Playback (WAV) =====
    private fun startPlayback() {
        val ok = outputWavFile.exists() && outputWavFile.length() > 44
        Log.d(TAG, "startPlayback() exists=$ok size=${outputWavFile.length()}")
        if (!ok) return

        player?.release()
        player = MediaPlayer().apply {
            setDataSource(outputWavFile.absolutePath)
            setOnPreparedListener { it.start() }
            setOnCompletionListener {
                it.release()
                player = null
                Log.d(TAG, "Playback complete")
            }
            try {
                prepareAsync()
            } catch (t: Throwable) {
                Log.e(TAG, "MediaPlayer.prepareAsync failed", t)
                runCatching { release() }
                player = null
            }
        }
    }

    private fun stopPlayback() {
        player?.runCatching { stop(); release() }?.onFailure { Log.w(TAG, "stopPlayback failed", it) }
        player = null
    }

    // Für die Wellenform: aktuelle Amplitude (0..32767)
    private fun getRecorderAmplitude(): Int = if (isRecordingFlag) lastAmplitude else 0

    // ===== Whisper: Model laden + Transkription starten =====
    private suspend fun ensureWhisperCtx(): Long = withContext(Dispatchers.IO) {
        if (whisperCtx != 0L) return@withContext whisperCtx

        val modelFile = copyAssetIfNeeded(modelAssetPath)
        Log.i(TAG, "Model copied/ready at ${modelFile.absolutePath} size=${modelFile.length()}")
        require(modelFile.exists() && modelFile.length() > 1_000_000) {
            "Modell fehlt oder ist zu klein: ${modelFile.absolutePath}"
        }

        val t0 = System.currentTimeMillis()
        val ctx = try {
            NativeWhisper.init(modelFile.absolutePath)
        } catch (t: Throwable) {
            Log.e(TAG, "NativeWhisper.init exception", t)
            0L
        }
        val dt = System.currentTimeMillis() - t0
        Log.i(TAG, "NativeWhisper.init returned=$ctx in ${dt}ms")

        check(ctx != 0L) { "Whisper-Initialisierung fehlgeschlagen (siehe Logcat)" }
        whisperCtx = ctx
        ctx
    }

    private fun onTranscribeRequested() {
        if (!outputWavFile.exists() || outputWavFile.length() <= 44) {
            Log.w(TAG, "Transcribe requested but WAV missing/too small (size=${outputWavFile.length()})")
            _setTranscribeState?.invoke(false, "WAV fehlt oder ist leer.", null)
            return
        }

        Log.i(TAG, "Transcribe requested size=${outputWavFile.length()} path=${outputWavFile.absolutePath}")

        lifecycleScope.launch {
            _setTranscribeState?.invoke(true, null, null)

            val result: Result<String> = runCatching {
                ensureWhisperCtx()

                val wav = readWavMono16(outputWavFile)
                Log.d(TAG, "WAV parsed: sr=${wav.sampleRate} samples=${wav.mono16.size}")
                require(wav.sampleRate == SAMPLE_RATE) { "WAV muss 16 kHz sein" }
                val pcmF32 = shortToFloatMono(wav.mono16)

                // deutliche Logs + Timeout um das JNI einzugrenzen
                withContext(Dispatchers.Default) {
                    withTimeout(30_000) {
                        val t0 = System.currentTimeMillis()
                        Log.i(TAG, "Calling fullTranscribe: ctx=$whisperCtx, samples=${pcmF32.size}, sr=$SAMPLE_RATE")
                        val outStr = NativeWhisper.fullTranscribe(whisperCtx, pcmF32, SAMPLE_RATE)
                        Log.i(TAG, "fullTranscribe done len=${outStr.length} in ${System.currentTimeMillis() - t0}ms")
                        outStr
                    }
                }
            }

            result.onSuccess { value ->
                val out = value.trim()
                if (out.isEmpty()) {
                    Log.w(TAG, "Transcription returned empty text")
                    _setTranscribeState?.invoke(false, "Leeres Ergebnis vom Transcriber.", null)
                } else {
                    _setTranscribeState?.invoke(false, null, out)
                }
            }.onFailure { e ->
                Log.e(TAG, "Transcription failed", e)
                _setTranscribeState?.invoke(false, e.message ?: "Transkription fehlgeschlagen", null)
            }
        }
    }


    // WAV-Reader (Mono 16-bit PCM)
    data class WavPcm(val sampleRate: Int, val mono16: ShortArray)

    private fun readWavMono16(file: File): WavPcm {
        val data = file.readBytes()
        fun le16(off: Int) = (data[off].toInt() and 0xFF) or ((data[off+1].toInt() and 0xFF) shl 8)
        fun le32(off: Int) = le16(off) or (le16(off+2) shl 16)

        require(String(data, 0, 4) == "RIFF" && String(data, 8, 4) == "WAVE") { "Kein WAV" }

        var pos = 12
        var sampleRate = SAMPLE_RATE
        var numChannels = 1
        var bitsPerSample = 16
        var dataStart = -1
        var dataLen = -1

        while (pos + 8 <= data.size) {
            val id = String(data, pos, 4)
            val size = le32(pos + 4)
            when (id) {
                "fmt " -> {
                    val audioFormat = le16(pos + 8) // 1 = PCM
                    numChannels = le16(pos + 10)
                    sampleRate = le32(pos + 12)
                    bitsPerSample = le16(pos + 22)
                    require(audioFormat == 1) { "Nicht PCM" }
                }
                "data" -> {
                    dataStart = pos + 8
                    dataLen = size
                    break
                }
            }
            pos += 8 + size
        }
        require(dataStart >= 0) { "Kein data-Chunk" }
        require(bitsPerSample == 16 && numChannels == 1) { "Erwarte Mono 16-bit" }

        val samples = ShortArray(dataLen / 2)
        ByteBuffer.wrap(data, dataStart, dataLen)
            .order(ByteOrder.LITTLE_ENDIAN)
            .asShortBuffer()
            .get(samples)

        return WavPcm(sampleRate, samples)
    }

    private fun shortToFloatMono(x: ShortArray): FloatArray =
        FloatArray(x.size) { i -> (x[i] / 32768.0f).coerceIn(-1f, 1f) }

    // Modell aus Assets in internen Speicher kopieren (einmalig)
    private suspend fun copyAssetIfNeeded(assetName: String): File = withContext(Dispatchers.IO) {
        val outFile = File(filesDir, assetName.substringAfterLast('/'))
        if (outFile.exists() && outFile.length() > 1_000_000) {
            Log.d(TAG, "Model already present: ${outFile.absolutePath} (${outFile.length()} bytes)")
            return@withContext outFile
        }
        Log.i(TAG, "Copying asset '$assetName' to ${outFile.absolutePath}")
        assets.open(assetName).use { input ->
            outFile.outputStream().use { output -> input.copyTo(output) }
        }
        Log.i(TAG, "Model copied size=${outFile.length()}")
        outFile
    }

    override fun onStop() {
        super.onStop()
        stopPcmRecording()
        stopPlayback()
    }

    override fun onDestroy() {
        super.onDestroy()
        if (whisperCtx != 0L) {
            runCatching { NativeWhisper.free(whisperCtx) }
                .onFailure { Log.w(TAG, "NativeWhisper.free failed", it) }
            whisperCtx = 0L
            Log.i(TAG, "Whisper context freed")
        }
    }

    // --- Compose ↔ Activity: Callback, um UI-States von hier setzen zu können ---
    private var _setTranscribeState: ((Boolean, String?, String?) -> Unit)? = null
    fun registerTranscribeStateSetter(setter: (Boolean, String?, String?) -> Unit) {
        _setTranscribeState = setter
    }
}

@OptIn(ExperimentalComposeUiApi::class)
@Composable
private fun RecorderScreen(
    isMicGranted: () -> Boolean,
    askForMic: () -> Unit,
    startRecording: () -> Unit,
    stopRecording: () -> Unit,
    startPlayback: () -> Unit,
    stopPlayback: () -> Unit,
    hasRecording: () -> Boolean,
    getAmplitude: () -> Int,
    onTranscribe: () -> Unit,
) {
    val activity = LocalContext.current as MainActivity

    var isRecording by remember { mutableStateOf(false) }
    var recordingReady by remember { mutableStateOf(false) }
    var lastError by remember { mutableStateOf<String?>(null) }

    // Transkript-UI-State
    var isTranscribing by remember { mutableStateOf(false) }
    var transcript by remember { mutableStateOf<String?>(null) }
    var transcribeError by remember { mutableStateOf<String?>(null) }

    // Timer-State (Millis)
    var elapsedMs by remember { mutableStateOf(0L) }

    // Waveform-State
    val amplitudes = remember { mutableStateListOf<Float>() }
    val maxSamples = 120

    // Initial
    LaunchedEffect(Unit) {
        recordingReady = hasRecording()
        activity.registerTranscribeStateSetter { loading, err, text ->
            isTranscribing = loading
            transcribeError = err
            transcript = text
        }
    }

    // Während Aufnahme: Timer & Amplitude samplen
    LaunchedEffect(isRecording) {
        if (isRecording) {
            val startTime = System.currentTimeMillis()
            while (isRecording) {
                elapsedMs = System.currentTimeMillis() - startTime
                val normalized = (getAmplitude().coerceAtLeast(0) / 32767f).coerceIn(0f, 1f)
                amplitudes.add(normalized)
                if (amplitudes.size > maxSamples) amplitudes.removeAt(0)
                delay(40)
            }
        }
    }

    val canPlay by remember { derivedStateOf { recordingReady && !isRecording && !isTranscribing } }
    val canRecord by remember { derivedStateOf { !isTranscribing } }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp, Alignment.Top),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            "Simple Voice Recorder (WAV 16 kHz) + Whisper",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold
        )

        // Statuszeile
        Text(
            text = when {
                isRecording     -> "Aufnahme: ${formatTime(elapsedMs)}"
                isTranscribing  -> "Transkribiere… bitte warten"
                recordingReady  -> "Letzte Aufnahme bereit • Play oder Transkribieren"
                else            -> "Noch keine Aufnahme"
            },
            style = MaterialTheme.typography.bodyLarge
        )

        // Wellenform-Anzeige
        Waveform(
            amplitudes = amplitudes,
            barWidth = 4.dp,
            barGap = 2.dp,
            height = 96.dp
        )

        // RECORD Button (press & hold)
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(64.dp)
                .background(
                    color = if (isRecording) MaterialTheme.colorScheme.errorContainer
                    else MaterialTheme.colorScheme.primaryContainer,
                    shape = RoundedCornerShape(16.dp)
                )
                .pointerInteropFilter { event ->
                    if (!canRecord) return@pointerInteropFilter true
                    when (event.action) {
                        MotionEvent.ACTION_DOWN -> {
                            if (!isMicGranted()) {
                                askForMic()
                                lastError = "Mikrofon-Zugriff benötigt. Erlaube die Berechtigung und drücke erneut."
                            } else {
                                runCatching { startRecording() }
                                    .onSuccess {
                                        isRecording = true
                                        recordingReady = false
                                        lastError = null
                                        elapsedMs = 0L
                                        amplitudes.clear()
                                        transcript = null
                                        transcribeError = null
                                    }
                                    .onFailure {
                                        lastError = it.localizedMessage ?: "Aufnahme konnte nicht gestartet werden."
                                    }
                            }
                            true
                        }
                        MotionEvent.ACTION_UP,
                        MotionEvent.ACTION_CANCEL -> {
                            if (isRecording) {
                                runCatching { stopRecording() }
                                isRecording = false
                                recordingReady = hasRecording()
                            }
                            true
                        }
                        else -> false
                    }
                }
                .padding(horizontal = 16.dp),
            contentAlignment = Alignment.Center
        ) {
            Text(if (isRecording) "Recording… (Finger halten)" else "Record (zum Aufnehmen halten)")
        }

        // PLAY
        Button(
            onClick = { if (canPlay) startPlayback() },
            enabled = canPlay,
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp),
            shape = RoundedCornerShape(16.dp)
        ) { Text("Play") }

        // TRANSCRIBE
        Button(
            onClick = { if (recordingReady && !isTranscribing) onTranscribe() },
            enabled = recordingReady && !isTranscribing && !isRecording,
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp),
            shape = RoundedCornerShape(16.dp)
        ) { Text("Transkribieren") }

        // STOP PLAYBACK
        Button(
            onClick = { stopPlayback() },
            enabled = !isRecording,
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp),
            shape = RoundedCornerShape(16.dp)
        ) { Text("Stop") }

        // Fehler
        if (lastError != null) {
            Text(text = lastError!!, color = MaterialTheme.colorScheme.error)
        }
        if (transcribeError != null) {
            Text(text = transcribeError!!, color = MaterialTheme.colorScheme.error)
        }

        // Transkript-Ausgabe
        if (!transcript.isNullOrBlank()) {
            Spacer(Modifier.height(8.dp))
            Text(
                text = transcript!!,
                style = MaterialTheme.typography.bodyLarge
            )
        }
    }
}

@Composable
private fun Waveform(
    amplitudes: List<Float>,
    barWidth: Dp,
    barGap: Dp,
    height: Dp,
) {
    val barW = barWidth
    val gap = barGap
    val totalBarWidthPx = with(androidx.compose.ui.platform.LocalDensity.current) { (barW + gap).toPx() }
    val barWidthPx = with(androidx.compose.ui.platform.LocalDensity.current) { barW.toPx() }
    val color = MaterialTheme.colorScheme.primary

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(height)
            .background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(12.dp))
            .padding(horizontal = 8.dp, vertical = 8.dp)
    ) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val h = size.height
            val w = size.width
            if (amplitudes.isEmpty()) return@Canvas

            val maxBars = (w / totalBarWidthPx).roundToInt().coerceAtLeast(1)
            val data = if (amplitudes.size > maxBars) amplitudes.takeLast(maxBars) else amplitudes

            var x = w - barWidthPx
            for (i in data.indices.reversed()) {
                val amp = data[i].coerceIn(0f, 1f)
                val barHeight = (h * amp).coerceAtLeast(2f)
                val top = (h - barHeight) / 2f
                drawRect(
                    color = color,
                    topLeft = androidx.compose.ui.geometry.Offset(x, top),
                    size = androidx.compose.ui.geometry.Size(barWidthPx, barHeight)
                )
                x -= totalBarWidthPx
                if (x < 0f) break
            }
        }
    }
}

private fun formatTime(ms: Long): String {
    val totalSeconds = ms / 1000
    val minutes = totalSeconds / 60
    val seconds = totalSeconds % 60
    return "%d:%02d".format(minutes, seconds)
}

@Preview(showBackground = true)
@Composable
private fun RecorderScreenPreview() {
    WhispersyncTheme {
        RecorderScreen(
            isMicGranted = { true },
            askForMic = {},
            startRecording = {},
            stopRecording = {},
            startPlayback = {},
            stopPlayback = {},
            hasRecording = { true },
            getAmplitude = { (0..32767).random() },
            onTranscribe = {}
        )
    }
}
