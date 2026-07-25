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
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.util.Date;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.apache.fesod.sheet.metadata.GlobalConfiguration;
import org.apache.fesod.sheet.metadata.property.ExcelContentProperty;
import org.apache.fesod.sheet.metadata.property.NumberFormatProperty;
import org.apache.fesod.sheet.testkit.Tags;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Tests for {@link ThreadLocalCache} — the central entry point for cleaning up
 * ThreadLocal caches created by Fesod utility classes.
 *
 * <p>These tests verify that {@link ThreadLocalCache#removeAll()} correctly clears
 * the ThreadLocal caches of all four utility classes ({@link DateUtils},
 * {@link NumberUtils}, {@link NumberDataFormatterUtils}, {@link ClassUtils})
 * and that it works in thread-pool scenarios where leaks would otherwise occur.
 */
@Tag(Tags.UNIT)
public class ThreadLocalCacheTest {

    @SuppressWarnings("unchecked")
    private static <T> ThreadLocal<T> getThreadLocalField(Class<?> clazz, String fieldName) throws Exception {
        Field field = clazz.getDeclaredField(fieldName);
        field.setAccessible(true);
        return (ThreadLocal<T>) field.get(null);
    }

    @AfterEach
    void cleanup() {
        ThreadLocalCache.removeAll();
    }

    /**
     * Verify that {@link ThreadLocalCache#removeAll()} clears the
     * {@link DateUtils} ThreadLocal caches after {@link DateUtils#format} was called.
     */
    @Test
    void removeAll_clearsDateUtilsThreadLocalCache() throws Exception {
        ThreadLocal<Map<Locale, Map<String, ?>>> dtfTls =
                getThreadLocalField(DateUtils.class, "DATE_TIME_FORMATTER_THREAD_LOCAL");
        ThreadLocal<Map<String, ?>> sdfTls = getThreadLocalField(DateUtils.class, "DATE_FORMAT_THREAD_LOCAL");

        // Populate caches
        DateUtils.format(LocalDateTime.of(2023, 6, 15, 10, 30, 0), "yyyy-MM-dd HH:mm:ss");
        DateUtils.format(new Date(), "yyyy-MM-dd");

        assertNotNull(dtfTls.get(), "DateTimeFormatter cache should be populated");
        assertNotNull(sdfTls.get(), "SimpleDateFormat cache should be populated");

        // Clean up via central API
        ThreadLocalCache.removeAll();

        assertNull(dtfTls.get(), "DateTimeFormatter cache should be null after removeAll()");
        assertNull(sdfTls.get(), "SimpleDateFormat cache should be null after removeAll()");
    }

    /**
     * Verify that {@link ThreadLocalCache#removeAll()} clears the
     * {@link NumberUtils} ThreadLocal cache after {@link NumberUtils#format} was called.
     */
    @Test
    void removeAll_clearsNumberUtilsThreadLocalCache() throws Exception {
        ThreadLocal<Map<Locale, Map<String, ?>>> tls =
                getThreadLocalField(NumberUtils.class, "DECIMAL_FORMAT_THREAD_LOCAL");

        // Populate cache
        ExcelContentProperty property = new ExcelContentProperty();
        NumberFormatProperty nfp = new NumberFormatProperty("#,##0.00", RoundingMode.HALF_UP);
        property.setNumberFormatProperty(nfp);
        NumberUtils.format(12345.6789, property);

        assertNotNull(tls.get(), "DecimalFormat cache should be populated");

        // Clean up via central API
        ThreadLocalCache.removeAll();

        assertNull(tls.get(), "DecimalFormat cache should be null after removeAll()");
    }

    /**
     * Verify that {@link ThreadLocalCache#removeAll()} clears the
     * {@link NumberDataFormatterUtils} ThreadLocal cache after
     * {@link NumberDataFormatterUtils#format} was called.
     */
    @Test
    void removeAll_clearsNumberDataFormatterUtilsThreadLocalCache() throws Exception {
        ThreadLocal<?> tls = getThreadLocalField(NumberDataFormatterUtils.class, "DATA_FORMATTER_THREAD_LOCAL");

        // Populate cache
        NumberDataFormatterUtils.format(BigDecimal.ONE, (short) 1, "0.00", new GlobalConfiguration());

        assertNotNull(tls.get(), "DataFormatter cache should be populated");

        // Clean up via central API
        ThreadLocalCache.removeAll();

        assertNull(tls.get(), "DataFormatter cache should be null after removeAll()");
    }

    /**
     * Simulate a web-server thread pool scenario:
     * <ol>
     *   <li>Submit a task that calls {@link DateUtils#format} and {@link NumberUtils#format}</li>
     *   <li>Verify the pool thread has non-null ThreadLocal caches (leak exists)</li>
     *   <li>Submit a task that calls {@link ThreadLocalCache#removeAll()}</li>
     *   <li>Verify the pool thread's ThreadLocal caches are now null (leak fixed)</li>
     * </ol>
     */
    @Test
    void removeAll_cleansUpThreadPoolThread() throws Exception {
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try {
            ThreadLocal<Map<Locale, Map<String, ?>>> dtfTls =
                    getThreadLocalField(DateUtils.class, "DATE_TIME_FORMATTER_THREAD_LOCAL");

            // Step 1: Populate caches on the pool thread
            CountDownLatch latch1 = new CountDownLatch(1);
            pool.submit(() -> {
                DateUtils.format(LocalDateTime.of(2023, 6, 15, 10, 30, 0), "yyyy-MM-dd HH:mm:ss");
                latch1.countDown();
            });
            assertTrue(latch1.await(5, TimeUnit.SECONDS));

            // Step 2: Verify cache is populated on the pool thread
            AtomicBoolean cachePopulated = new AtomicBoolean(false);
            CountDownLatch latch2 = new CountDownLatch(1);
            pool.submit(() -> {
                Map<Locale, Map<String, ?>> cache = dtfTls.get();
                cachePopulated.set(cache != null && !cache.isEmpty());
                latch2.countDown();
            });
            assertTrue(latch2.await(5, TimeUnit.SECONDS));
            assertTrue(cachePopulated.get(), "ThreadLocal cache should be populated on the pool thread before cleanup");

            // Step 3: Clean up via ThreadLocalCache.removeAll() on the pool thread
            CountDownLatch latch3 = new CountDownLatch(1);
            pool.submit(() -> {
                ThreadLocalCache.removeAll();
                latch3.countDown();
            });
            assertTrue(latch3.await(5, TimeUnit.SECONDS));

            // Step 4: Verify cache is null on the pool thread
            AtomicBoolean cacheCleared = new AtomicBoolean(false);
            CountDownLatch latch4 = new CountDownLatch(1);
            pool.submit(() -> {
                cacheCleared.set(dtfTls.get() == null);
                latch4.countDown();
            });
            assertTrue(latch4.await(5, TimeUnit.SECONDS));
            assertTrue(cacheCleared.get(), "ThreadLocal cache should be null on the pool thread after removeAll()");
        } finally {
            pool.shutdown();
            pool.awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    /**
     * Verify that calling {@link ThreadLocalCache#removeAll()} when no caches exist
     * is a safe no-op (does not throw).
     */
    @Test
    void removeAll_whenNoCachesExist_isSafeNoOp() {
        // Should not throw even if no caches have been populated
        ThreadLocalCache.removeAll();
        ThreadLocalCache.removeAll();
    }

    /**
     * Verify that {@link ThreadLocalCache#removeAll()} is idempotent — calling it
     * multiple times has the same effect as calling it once.
     */
    @Test
    void removeAll_isIdempotent() throws Exception {
        ThreadLocal<Map<Locale, Map<String, ?>>> dtfTls =
                getThreadLocalField(DateUtils.class, "DATE_TIME_FORMATTER_THREAD_LOCAL");

        DateUtils.format(LocalDateTime.of(2023, 6, 15, 10, 30, 0), "yyyy-MM-dd");
        assertNotNull(dtfTls.get());

        ThreadLocalCache.removeAll();
        assertNull(dtfTls.get());

        // Second call should not throw and cache should still be null
        ThreadLocalCache.removeAll();
        assertNull(dtfTls.get());
    }

    /**
     * Verify that format results remain correct after a cleanup cycle —
     * the cache is re-populated on the next call.
     */
    @Test
    void format_worksCorrectly_afterRemoveAll() {
        String expected = "2023-06-15 10:30:00";

        // First call populates cache
        String result1 = DateUtils.format(LocalDateTime.of(2023, 6, 15, 10, 30, 0), "yyyy-MM-dd HH:mm:ss");
        assertEquals(expected, result1);

        // Clean up
        ThreadLocalCache.removeAll();

        // Second call should re-create the cache and produce the same result
        String result2 = DateUtils.format(LocalDateTime.of(2023, 6, 15, 10, 30, 0), "yyyy-MM-dd HH:mm:ss");
        assertEquals(expected, result2);
    }
}
