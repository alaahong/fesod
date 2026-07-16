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

package org.apache.fesod.sheet.head;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.fesod.sheet.ExcelWriter;
import org.apache.fesod.sheet.FesodSheet;
import org.apache.fesod.sheet.testkit.Tags;
import org.apache.fesod.sheet.testkit.assertions.ExcelAssertions;
import org.apache.fesod.sheet.testkit.base.AbstractExcelTest;
import org.apache.fesod.sheet.testkit.builders.TestDataBuilder;
import org.apache.fesod.sheet.testkit.enums.ExcelFormat;
import org.apache.fesod.sheet.testkit.params.ExcelFormatSource;
import org.apache.fesod.sheet.write.metadata.WriteSheet;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;

/**
 * Test complex head write/read for all Excel formats using parameterized tests.
 */
@Tag(Tags.ROUND_TRIP)
public class ComplexHeadDataTest extends AbstractExcelTest {

    @ParameterizedTest
    @ExcelFormatSource
    void readAndWrite(ExcelFormat format) throws Exception {
        File file = createTempFile("complexHead", format);
        FesodSheet.write(file, ComplexHeadData.class).sheet().doWrite(TestDataBuilder.complexHeadData(1));
        List<ComplexHeadData> result = FesodSheet.read(file)
                .head(ComplexHeadData.class)
                .xlsxSAXParserFactoryName("com.sun.org.apache.xerces.internal.jaxp.SAXParserFactoryImpl")
                .sheet()
                .doReadSync();
        Assertions.assertEquals(1, result.size());
        Assertions.assertEquals("String4", result.get(0).getString4());
    }

    @ParameterizedTest
    @ExcelFormatSource
    void readAndWriteAutomaticMergeHead(ExcelFormat format) throws Exception {
        File file = createTempFile("complexHeadAutoMerge", format);
        FesodSheet.write(file, ComplexHeadData.class)
                .automaticMergeHead(Boolean.FALSE)
                .sheet()
                .doWrite(TestDataBuilder.complexHeadData(1));
        List<ComplexHeadData> result =
                FesodSheet.read(file).head(ComplexHeadData.class).sheet().doReadSync();
        Assertions.assertEquals(1, result.size());
        Assertions.assertEquals("String4", result.get(0).getString4());
    }

    /**
     * Real-file integration test: writes two real XLSX files using the same
     * {@code WriteSheet} configured via {@code head(Consumer)} with uneven
     * column depths, then reads both files back to verify headers and data
     * are correct in both.
     * <p>
     * The deep-copy fix ensures the {@code DefaultHeadBuilder}'s internal list
     * is not shared with the stored head, preventing silent state corruption
     * when the builder is reused.
     */
    @Test
    void headConsumer_reuseWriteSheet_producesCorrectFiles() throws Exception {
        File file1 = createTempFile("head-consumer-reuse-1", ExcelFormat.XLSX);
        File file2 = createTempFile("head-consumer-reuse-2", ExcelFormat.XLSX);

        // head(Consumer) with uneven column depths:
        //   column 0: ["ID"]            (depth 1, will be padded to 2)
        //   column 1: ["Info","Name"]   (depth 2)
        //   column 2: ["Info","Age"]    (depth 2)
        WriteSheet writeSheet = FesodSheet.writerSheet()
                .head(b -> b.column("ID")
                        .columns("Info", sub -> sub.column("Name").column("Age")))
                .build();

        // Prepare row data (no model class)
        List<List<Object>> data = new ArrayList<>();
        data.add(Arrays.asList(1, "Jackson", 20));
        data.add(Arrays.asList(2, "Tom", 21));

        // Write file 1
        try (ExcelWriter writer1 = FesodSheet.write(file1)
                .excelType(ExcelFormat.XLSX.toExcelTypeEnum())
                .build()) {
            writer1.write(data, writeSheet);
        }

        // Write file 2 with the same WriteSheet
        try (ExcelWriter writer2 = FesodSheet.write(file2)
                .excelType(ExcelFormat.XLSX.toExcelTypeEnum())
                .build()) {
            writer2.write(data, writeSheet);
        }

        // ---- Verify file 1 headers and data via real file I/O ----
        try (ExcelAssertions ea = ExcelAssertions.assertThat(file1)) {
            ea.sheet(0).hasRowCount(4); // 2 header rows + 2 data rows
            // Header row 0
            ea.sheet(0).row(0).cell(0).hasStringValue("ID");
            ea.sheet(0).row(0).cell(1).hasStringValue("Info");
            ea.sheet(0).row(0).cell(2).hasStringValue("Info");
            // Header row 1
            ea.sheet(0).row(1).cell(0).hasStringValue("ID");
            ea.sheet(0).row(1).cell(1).hasStringValue("Name");
            ea.sheet(0).row(1).cell(2).hasStringValue("Age");
            // Data row 0
            ea.sheet(0).row(2).cell(0).hasNumericValue(1.0);
            ea.sheet(0).row(2).cell(1).hasStringValue("Jackson");
            ea.sheet(0).row(2).cell(2).hasNumericValue(20.0);
        }

        // ---- Verify file 2 headers and data (must match file 1) ----
        try (ExcelAssertions ea = ExcelAssertions.assertThat(file2)) {
            ea.sheet(0).hasRowCount(4);
            ea.sheet(0).row(0).cell(0).hasStringValue("ID");
            ea.sheet(0).row(0).cell(1).hasStringValue("Info");
            ea.sheet(0).row(0).cell(2).hasStringValue("Info");
            ea.sheet(0).row(1).cell(0).hasStringValue("ID");
            ea.sheet(0).row(1).cell(1).hasStringValue("Name");
            ea.sheet(0).row(1).cell(2).hasStringValue("Age");
            ea.sheet(0).row(2).cell(0).hasNumericValue(1.0);
            ea.sheet(0).row(2).cell(1).hasStringValue("Jackson");
            ea.sheet(0).row(2).cell(2).hasNumericValue(20.0);
        }
    }
}
