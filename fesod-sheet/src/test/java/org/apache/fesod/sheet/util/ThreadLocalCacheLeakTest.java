/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.fesod.sheet.util;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import java.lang.reflect.Field;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.apache.fesod.sheet.testkit.Tags;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Reproduces the ThreadLocal cache leak when {@link DateUtils} public static API is called
 * directly (outside of a Fesod read/write flow).
 *
 * <p>The {@code removeThreadLocalCache()} cleanup is only invoked at the end of
 * {@code WriteContextImpl.finish()} and {@code ExcelAnalyserImpl} (the read flow).
 * If a user calls {@code DateUtils.format(...)} directly — e.g. in a web-server thread
 * pool for data transformation — the ThreadLocal cache is never cleaned up, causing
 * a slow memory leak in pooled-thread scenarios.
 */
@Tag(Tags.UNIT)
public class ThreadLocalCacheLeakTest {

    /**
     * Reflective access to the private static ThreadLocal fields so we can inspect
     * their state without relying on the public cleanup API.
     */
    @SuppressWarnings("unchecked")
    private static <T> ThreadLocal<T> getThreadLocalField(Class<?> clazz, String fieldName) throws Exception {
        Field field = clazz.getDeclaredField(fieldName);
        field.setAccessible(true);
        return (ThreadLocal<T>) field.get(null);
    }

    @AfterEach
    void cleanup() {
        // Ensure we don't pollute other tests
        DateUtils.removeThreadLocalCache();
        NumberUtils.removeThreadLocalCache();
    }

    /**
     * Proof 1: Calling {@link DateUtils#format(LocalDate, String)} directly leaves a
     * non-null entry in the {@code DATE_TIME_FORMATTER_THREAD_LOCAL}.
     *
     * <p>Before the call, the ThreadLocal is null. After the call, it holds a cache map.
     * Without calling {@link DateUtils#removeThreadLocalCache()}, it stays forever.
     */
    @Test
    void dateUtilsFormat_directCall_leavesThreadLocalResidue() throws Exception {
        ThreadLocal<Map<Locale, Map<String, ?>>> tls =
                getThreadLocalField(DateUtils.class, "DATE_TIME_FORMATTER_THREAD_LOCAL");

        // Before: no cache on this thread
        assertNull(tls.get(), "ThreadLocal should be null before any format() call");

        // Direct API call — simulates a user calling DateUtils.format() in a web handler
        String result = DateUtils.format(LocalDate.of(2023, 6, 15), "yyyy-MM-dd");
        assertEquals("2023-06-15", result);

        // After: cache is non-null and NOT cleaned up
        Map<Locale, Map<String, ?>> cache = tls.get();
        assertNotNull(cache, "ThreadLocal should be non-null after format() call");
        assertFalse_cacheIsEmpty(cache);
    }

    /**
     * Proof 2: Multiple different format patterns accumulate entries in the ThreadLocal
     * cache. Each unique pattern creates a new DateTimeFormatter that stays in memory
     * until removeThreadLocalCache() is called.
     */
    @Test
    void dateUtilsFormat_multiplePatterns_accumulateInThreadLocal() throws Exception {
        ThreadLocal<Map<Locale, Map<String, ?>>> tls =
                getThreadLocalField(DateUtils.class, "DATE_TIME_FORMATTER_THREAD_LOCAL");

        assertNull(tls.get(), "ThreadLocal should be null at start");

        // Simulate a web service handling requests with different date formats.
        // Use LocalDateTime for all patterns since it supports both date and time fields.
        String[] patterns = {
            "yyyy-MM-dd", "yyyy/MM/dd", "dd-MM-yyyy", "MM/dd/yyyy",
            "yyyy-MM-dd HH:mm", "yyyy-MM-dd HH:mm:ss", "yyyyMMdd", "dd/MM/yyyy"
        };

        for (String pattern : patterns) {
            DateUtils.format(LocalDateTime.of(2023, 6, 15, 10, 30, 0), pattern);
        }

        Map<Locale, Map<String, ?>> cache = tls.get();
        assertNotNull(cache, "ThreadLocal cache should exist after multiple calls");

        // Count total cached formatters across all locales
        int totalFormatters = cache.values().stream().mapToInt(Map::size).sum();
        assertEquals(
                patterns.length,
                totalFormatters,
                "Each unique pattern should create one cached formatter, got: " + totalFormatters);

        // After removeThreadLocalCache, the ThreadLocal is cleared
        DateUtils.removeThreadLocalCache();
        assertNull(tls.get(), "ThreadLocal should be null after removeThreadLocalCache()");
    }

    /**
     * Proof 3: Thread pool simulation — a task that calls DateUtils.format() directly
     * leaves residue on the pool thread even after the task completes.
     *
     * <p>This is the core leak scenario: in a web server (Tomcat, Jetty, etc.), threads
     * are reused across requests. Each request that calls DateUtils.format() adds to
     * the ThreadLocal without ever cleaning up.
     */
    @Test
    void threadLocalCache_leaksAcrossThreadPoolTasks() throws Exception {
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try {
            ThreadLocal<Map<Locale, Map<String, ?>>> tls =
                    getThreadLocalField(DateUtils.class, "DATE_TIME_FORMATTER_THREAD_LOCAL");

            // Step 1: Submit a task that calls DateUtils.format() directly
            AtomicReference<Long> threadId = new AtomicReference<>();
            CountDownLatch latch1 = new CountDownLatch(1);
            pool.submit(() -> {
                threadId.set(Thread.currentThread().getId());
                DateUtils.format(LocalDate.of(2023, 6, 15), "yyyy-MM-dd");
                latch1.countDown();
            });
            assertTrue(latch1.await(5, TimeUnit.SECONDS));

            // Step 2: Verify the pool thread now has a non-null ThreadLocal
            // We can't directly read another thread's ThreadLocal, but we can check
            // that a subsequent task on the SAME thread sees the residue.
            AtomicReference<Boolean> residueFound = new AtomicReference<>(false);
            CountDownLatch latch2 = new CountDownLatch(1);
            pool.submit(() -> {
                // Same thread — ThreadLocal from previous task should still be here
                Map<Locale, Map<String, ?>> cache = tls.get();
                residueFound.set(cache != null && !cache.isEmpty());
                latch2.countDown();
            });
            assertTrue(latch2.await(5, TimeUnit.SECONDS));

            assertTrue(
                    residueFound.get(),
                    "ThreadLocal from a previous task should still be present on the reused pool thread — "
                            + "this proves the leak. Thread ID: " + threadId.get());

            // Step 3: Clean up the pool thread
            pool.submit(() -> DateUtils.removeThreadLocalCache()).get(5, TimeUnit.SECONDS);
        } finally {
            pool.shutdown();
            pool.awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    /**
     * Proof 4: The only way to clean up is to call {@link DateUtils#removeThreadLocalCache()}.
     * There is no automatic cleanup (no try-finally, no thread-exit hook registered by Fesod).
     * This test verifies that calling format() N times without cleanup leaves N cached
     * formatters, and that a single removeThreadLocalCache() clears everything.
     */
    @Test
    void removeThreadLocalCache_isTheOnlyCleanupMechanism() throws Exception {
        ThreadLocal<Map<Locale, Map<String, ?>>> tls =
                getThreadLocalField(DateUtils.class, "DATE_TIME_FORMATTER_THREAD_LOCAL");

        // Generate 5 different cached formatters.
        // Non-pattern text is wrapped in single quotes so it's treated as a literal,
        // not as pattern letters (e.g. 'p'=padding, 'a'=am-pm, 't'=unknown).
        String[] patterns = {
            "'fmt0'-yyyy-MM-dd", "'fmt1'-yyyy-MM-dd", "'fmt2'-yyyy-MM-dd", "'fmt3'-yyyy-MM-dd", "'fmt4'-yyyy-MM-dd"
        };
        for (String pattern : patterns) {
            DateUtils.format(LocalDate.of(2023, 1, 1), pattern);
        }

        Map<Locale, Map<String, ?>> cache = tls.get();
        assertNotNull(cache);
        int total = cache.values().stream().mapToInt(Map::size).sum();
        assertEquals(5, total, "Should have 5 cached formatters");

        // GC does NOT clean ThreadLocal (it's a strong reference held by the thread)
        System.gc();
        Thread.sleep(100);
        cache = tls.get();
        assertNotNull(cache, "ThreadLocal survives GC — it's a strong reference on the thread");
        assertEquals(
                5, cache.values().stream().mapToInt(Map::size).sum(), "Cache should still have 5 formatters after GC");

        // Only explicit cleanup works
        DateUtils.removeThreadLocalCache();
        assertNull(tls.get(), "Only removeThreadLocalCache() clears the ThreadLocal");
    }

    private static void assertFalse_cacheIsEmpty(Map<Locale, Map<String, ?>> cache) {
        boolean empty = cache.values().stream().allMatch(Map::isEmpty);
        assertTrue(!empty, "Cache should contain at least one formatter, but all locale maps are empty");
    }
}
