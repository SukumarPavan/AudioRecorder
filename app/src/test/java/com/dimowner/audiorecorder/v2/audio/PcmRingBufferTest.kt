/*
 * Copyright 2026 Dmytro Ponomarenko
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.dimowner.audiorecorder.v2.audio

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/**
 * Unit tests for [PcmRingBuffer]: frame alignment, snapshots before and after the first wrap,
 * snapshots taken while a writer keeps running, and the WAV header of a snapshot.
 *
 * Runs on real temporary files; no Android framework involved.
 */
class PcmRingBufferTest {

    @get:Rule
    val tempFolder = TemporaryFolder()

    private fun newBuffer(capacity: Long, frameSize: Int = 2): PcmRingBuffer =
        PcmRingBuffer(File(tempFolder.root, "ring.pcm"), capacity, frameSize)

    private fun sequentialBytes(from: Int, count: Int): ByteArray =
        ByteArray(count) { ((from + it) and 0xFF).toByte() }

    private fun File.pcmData(): ByteArray = readBytes().copyOfRange(WAV_HEADER_SIZE, length().toInt())

    private fun ByteArray.intLe(offset: Int): Int =
        ByteBuffer.wrap(this, offset, 4).order(ByteOrder.LITTLE_ENDIAN).int

    private fun ByteArray.shortLe(offset: Int): Int =
        ByteBuffer.wrap(this, offset, 2).order(ByteOrder.LITTLE_ENDIAN).short.toInt()

    // ---- frame alignment ----

    @Test
    fun `capacity is rounded down to whole frames`() {
        newBuffer(capacity = 1001, frameSize = 2).use { assertEquals(1000L, it.capacity) }
        newBuffer(capacity = 1003, frameSize = 4).use { assertEquals(1000L, it.capacity) }
    }

    @Test
    fun `odd sized writes only commit whole frames`() {
        newBuffer(capacity = 1000, frameSize = 2).use { buffer ->
            buffer.write(sequentialBytes(0, 3))
            assertEquals(2L, buffer.bufferedBytes)
            buffer.write(sequentialBytes(3, 1))
            assertEquals(4L, buffer.bufferedBytes)
            buffer.write(sequentialBytes(4, 5))
            assertEquals(8L, buffer.bufferedBytes)

            val out = tempFolder.newFile("aligned.wav")
            val dataLen = buffer.snapshotToWav(out)
            assertEquals(8L, dataLen)
            assertArrayEquals(sequentialBytes(0, 8), out.pcmData())
        }
    }

    @Test
    fun `split frames are reassembled across many writes and wraps`() {
        newBuffer(capacity = 64, frameSize = 4).use { buffer ->
            val all = sequentialBytes(0, 1003)
            var offset = 0
            var step = 1
            while (offset < all.size) {
                val n = minOf(step, all.size - offset)
                buffer.write(all, offset, n)
                assertEquals(0L, buffer.totalBytesWritten % 4)
                offset += n
                step = step % 7 + 1
            }
            // 1003 bytes = 250 whole frames + 3 pending bytes.
            assertEquals(1000L, buffer.totalBytesWritten)
            val out = tempFolder.newFile("reassembled.wav")
            assertEquals(64L, buffer.snapshotToWav(out))
            assertArrayEquals(all.copyOfRange(1000 - 64, 1000), out.pcmData())
        }
    }

    @Test
    fun `discarding a partial frame keeps later audio aligned`() {
        newBuffer(capacity = 100, frameSize = 2).use { buffer ->
            buffer.write(byteArrayOf(1, 2, 3))
            buffer.discardPartialFrame()
            buffer.write(byteArrayOf(10, 11))
            val out = tempFolder.newFile("discard.wav")
            buffer.snapshotToWav(out)
            assertArrayEquals(byteArrayOf(1, 2, 10, 11), out.pcmData())
        }
    }

    // ---- before the first wrap ----

    @Test
    fun `snapshot before the first wrap holds exactly what was written`() {
        newBuffer(capacity = 1000).use { buffer ->
            buffer.write(sequentialBytes(0, 300))
            assertFalse(buffer.hasWrapped)
            assertEquals(300L, buffer.bufferedBytes)

            val out = tempFolder.newFile("before-wrap.wav")
            assertEquals(300L, buffer.snapshotToWav(out))
            assertEquals(WAV_HEADER_SIZE + 300L, out.length())
            assertArrayEquals(sequentialBytes(0, 300), out.pcmData())
        }
    }

    @Test
    fun `snapshot of a buffer filled exactly to capacity is not treated as wrapped`() {
        newBuffer(capacity = 256).use { buffer ->
            buffer.write(sequentialBytes(0, 256))
            assertFalse(buffer.hasWrapped)
            val out = tempFolder.newFile("full.wav")
            assertEquals(256L, buffer.snapshotToWav(out))
            assertArrayEquals(sequentialBytes(0, 256), out.pcmData())
        }
    }

    @Test
    fun `snapshot of an empty buffer is a valid header with no data`() {
        newBuffer(capacity = 1000).use { buffer ->
            val out = tempFolder.newFile("empty.wav")
            assertEquals(0L, buffer.snapshotToWav(out))
            assertEquals(WAV_HEADER_SIZE.toLong(), out.length())
            val bytes = out.readBytes()
            assertEquals(36, bytes.intLe(4))
            assertEquals(0, bytes.intLe(40))
        }
    }

    // ---- after wrapping ----

    @Test
    fun `snapshot after wrapping returns the newest capacity bytes oldest first`() {
        newBuffer(capacity = 1000).use { buffer ->
            // 2.5 laps, written in uneven chunks so the head lands mid-file.
            val all = sequentialBytes(0, 2500)
            var offset = 0
            for (size in generateSequence(130) { it + 34 }) {
                if (offset >= all.size) break
                val n = minOf(size, all.size - offset)
                buffer.write(all, offset, n)
                offset += n
            }
            assertTrue(buffer.hasWrapped)
            assertEquals(1000L, buffer.bufferedBytes)

            val out = tempFolder.newFile("wrapped.wav")
            assertEquals(1000L, buffer.snapshotToWav(out))
            assertArrayEquals(all.copyOfRange(1500, 2500), out.pcmData())
        }
    }

    @Test
    fun `a single write larger than capacity keeps only its tail`() {
        newBuffer(capacity = 100).use { buffer ->
            buffer.write(sequentialBytes(0, 10))
            buffer.write(sequentialBytes(10, 350))
            assertEquals(360L, buffer.totalBytesWritten)
            val out = tempFolder.newFile("oversized.wav")
            assertEquals(100L, buffer.snapshotToWav(out))
            assertArrayEquals(sequentialBytes(260, 100), out.pcmData())
        }
    }

    @Test
    fun `repeated snapshots do not disturb the buffer`() {
        newBuffer(capacity = 500).use { buffer ->
            buffer.write(sequentialBytes(0, 800))
            val first = tempFolder.newFile("first.wav")
            buffer.snapshotToWav(first)
            buffer.write(sequentialBytes(800, 100))
            val second = tempFolder.newFile("second.wav")
            buffer.snapshotToWav(second)
            assertArrayEquals(sequentialBytes(300, 500), first.pcmData())
            assertArrayEquals(sequentialBytes(400, 500), second.pcmData())
        }
    }

    // ---- concurrent writes ----

    /**
     * Each 4-byte frame carries its own index, so a snapshot is correct if its frames are whole,
     * strictly increasing (in order, never torn) and end at or after where the snapshot began.
     */
    @Test
    fun `snapshots taken while the writer runs are whole frames in order`() {
        val frameSize = 4
        val capacityFrames = 50_000 // 200 KB: several snapshot chunks, many laps during the test
        newBuffer(capacity = capacityFrames.toLong() * frameSize, frameSize = frameSize).use { buffer ->
            val running = AtomicBoolean(true)
            val writerError = AtomicReference<Throwable?>(null)
            val started = CountDownLatch(1)
            val writer = Thread {
                try {
                    var index = 0
                    val chunk = ByteBuffer.allocate(frameSize * 317).order(ByteOrder.LITTLE_ENDIAN)
                    val bytes = chunk.array()
                    while (running.get()) {
                        chunk.clear()
                        while (chunk.remaining() >= frameSize) chunk.putInt(index++)
                        // Split at an odd offset so individual writes are not frame aligned.
                        val split = 4 * 101 + 3
                        buffer.write(bytes, 0, split)
                        buffer.write(bytes, split, bytes.size - split)
                        if (index > capacityFrames) started.countDown()
                    }
                } catch (t: Throwable) {
                    writerError.set(t)
                    started.countDown()
                }
            }
            writer.start()
            assertTrue("writer did not wrap in time", started.await(30, TimeUnit.SECONDS))

            repeat(20) { round ->
                val before = buffer.totalBytesWritten / frameSize
                val out = tempFolder.newFile("concurrent-$round.wav")
                val dataLen = buffer.snapshotToWav(out)
                assertEquals(0L, dataLen % frameSize)
                assertTrue(dataLen in 1..capacityFrames.toLong() * frameSize)

                val data = out.pcmData()
                assertEquals(dataLen.toInt(), data.size)
                var previous = data.intLe(0) - 1
                var i = 0
                while (i < data.size) {
                    val value = data.intLe(i)
                    assertTrue("frame $value after $previous is out of order", value > previous)
                    previous = value
                    i += frameSize
                }
                // The snapshot ends at the head as it was when it started.
                assertTrue(previous + 1 >= before)
                assertEquals(dataLen, out.length() - WAV_HEADER_SIZE)
                assertEquals(dataLen.toInt(), out.readBytes().intLe(40))
            }
            running.set(false)
            writer.join(5_000)
            writerError.get()?.let { throw AssertionError("writer failed", it) }
        }
    }

    // ---- WAV header ----

    @Test
    fun `snapshot header describes 16 kHz mono 16-bit PCM with correct sizes`() {
        newBuffer(capacity = ReplayBufferConfig.bufferSizeBytes(5)).use { buffer ->
            buffer.write(ByteArray(32_000) { 7 })
            val out = tempFolder.newFile("header.wav")
            val dataLen = buffer.snapshotToWav(out)
            val bytes = out.readBytes()

            assertEquals(32_000L, dataLen)
            assertEquals(WAV_HEADER_SIZE + 32_000, bytes.size)
            assertEquals("RIFF", String(bytes, 0, 4, Charsets.US_ASCII))
            assertEquals(32_000 + 36, bytes.intLe(4)) // RIFF chunk size = file size - 8
            assertEquals("WAVE", String(bytes, 8, 4, Charsets.US_ASCII))
            assertEquals("fmt ", String(bytes, 12, 4, Charsets.US_ASCII))
            assertEquals(16, bytes.intLe(16))
            assertEquals(1, bytes.shortLe(20)) // PCM
            assertEquals(1, bytes.shortLe(22)) // mono
            assertEquals(16_000, bytes.intLe(24))
            assertEquals(32_000, bytes.intLe(28)) // byte rate
            assertEquals(2, bytes.shortLe(32)) // block align
            assertEquals(16, bytes.shortLe(34)) // bits per sample
            assertEquals("data", String(bytes, 36, 4, Charsets.US_ASCII))
            assertEquals(32_000, bytes.intLe(40))
        }
    }

    @Test
    fun `header sizes match a wrapped buffer of the largest option`() {
        // 60 minutes: the biggest buffer a user can choose, still far under the 2 GiB WAV limit.
        val capacity = ReplayBufferConfig.bufferSizeBytes(60)
        assertEquals(115_200_000L, capacity)
        newBuffer(capacity = 4_000).use { buffer ->
            buffer.write(ByteArray(10_001))
            val out = tempFolder.newFile("wrapped-header.wav")
            assertEquals(4_000L, buffer.snapshotToWav(out))
            val bytes = out.readBytes()
            assertEquals(4_000 + 36, bytes.intLe(4))
            assertEquals(4_000, bytes.intLe(40))
            assertEquals(out.length() - 8, bytes.intLe(4).toLong())
        }
    }

    // ---- lifecycle ----

    @Test
    fun `a new buffer discards a stale file left behind`() {
        val file = File(tempFolder.root, "ring.pcm")
        file.writeBytes(ByteArray(500) { 9 })
        PcmRingBuffer(file, 1000).use { buffer ->
            assertEquals(0L, buffer.bufferedBytes)
            assertEquals(0L, file.length())
        }
    }

    @Test(expected = IOException::class)
    fun `writing to a closed buffer fails`() {
        val buffer = newBuffer(capacity = 100)
        buffer.close()
        buffer.write(ByteArray(4))
    }

    @Test(expected = IOException::class)
    fun `snapshotting a closed buffer fails`() {
        val buffer = newBuffer(capacity = 100)
        buffer.write(ByteArray(4))
        buffer.close()
        buffer.snapshotToWav(tempFolder.newFile("closed.wav"))
    }

    @Test
    fun `closeAndDelete removes the buffer file`() {
        val file = File(tempFolder.root, "ring.pcm")
        val buffer = PcmRingBuffer(file, 100)
        buffer.write(ByteArray(10))
        buffer.closeAndDelete()
        assertFalse(file.exists())
    }
}
