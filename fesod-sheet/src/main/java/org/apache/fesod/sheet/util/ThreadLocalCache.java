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

/*
 * This file is part of the Apache Fesod (Incubating) project, which was derived from Alibaba EasyExcel.
 *
 * Copyright (C) 2018-2024 Alibaba Group Holding Ltd.
 */

package org.apache.fesod.sheet.util;

/**
 * Central entry point for cleaning up ThreadLocal caches created by Fesod utility classes.
 *
 * <p>Fesod uses {@link ThreadLocal} caches in several utility classes to avoid repeatedly
 * constructing expensive formatter objects ({@link java.time.format.DateTimeFormatter},
 * {@link java.text.SimpleDateFormat}, {@link java.text.DecimalFormat}, etc.). When Fesod's
 * read or write flow completes, it automatically cleans up these caches. However, when
 * calling public utility methods directly (e.g. {@link DateUtils#format} or
 * {@link NumberUtils#format}), the caches are <strong>not</strong> automatically cleaned.
 *
 * <p>In thread-pool environments (e.g. web servers like Tomcat, Jetty), threads are reused
 * across requests. Each call to {@code DateUtils.format()} or {@code NumberUtils.format()}
 * populates the ThreadLocal cache, and these entries persist until the thread terminates
 * or {@link #removeAll()} is called explicitly. This can lead to unbounded memory growth
 * if many distinct format patterns are used.
 *
 * <p><strong>Usage example:</strong>
 * <pre>{@code
 * try {
 *     String formatted = DateUtils.format(LocalDateTime.now(), "yyyy-MM-dd HH:mm:ss");
 *     // ... more DateUtils / NumberUtils calls ...
 * } finally {
 *     ThreadLocalCache.removeAll();
 * }
 * }</pre>
 *
 * @see DateUtils#removeThreadLocalCache()
 * @see NumberUtils#removeThreadLocalCache()
 * @see NumberDataFormatterUtils#removeThreadLocalCache()
 * @see ClassUtils#removeThreadLocalCache()
 */
public final class ThreadLocalCache {

    private ThreadLocalCache() {}

    /**
     * Remove all ThreadLocal caches created by Fesod utility classes on the current thread.
     *
     * <p>This method is equivalent to calling:
     * <pre>{@code
     * DateUtils.removeThreadLocalCache();
     * NumberUtils.removeThreadLocalCache();
     * NumberDataFormatterUtils.removeThreadLocalCache();
     * ClassUtils.removeThreadLocalCache();
     * }</pre>
     *
     * <p>Call this method in a {@code finally} block after directly using Fesod utility
     * methods (e.g. {@link DateUtils#format}, {@link NumberUtils#format}) outside of
     * Fesod's read/write flow to prevent ThreadLocal memory leaks in thread-pool
     * environments.
     */
    public static void removeAll() {
        DateUtils.removeThreadLocalCache();
        NumberUtils.removeThreadLocalCache();
        NumberDataFormatterUtils.removeThreadLocalCache();
        ClassUtils.removeThreadLocalCache();
    }
}
