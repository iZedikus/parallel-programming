package ru.stepanov;

import lombok.extern.slf4j.Slf4j;
import org.apache.spark.sql.*;

import java.util.ArrayList;
import java.util.List;

import static org.apache.spark.sql.functions.*;

@Slf4j
public class BrooklynSalesAnalysis {

    public static void main(String[] args) {
        // 1. Создание SparkSession
        SparkSession spark = SparkSession.builder()
                .appName("BrooklynSalesAnalysis")
                .master("local[*]")  // для локального выполнения
                .getOrCreate();

        // Настройка уровня логирования
        spark.sparkContext().setLogLevel("WARN");
        log.info("Spark session created");

        // 2. Загрузка данных
        String filePath = "/Users/avrellian/Documents/Java Projects/parallel_programming/lab5/src/main/resources/brooklyn_sales_map.csv";
        Dataset<Row> df = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(filePath);

        log.info("Data loaded from {}", filePath);
        df.printSchema();
        df.show(5, false);

        // Задание 1: Добавить колонку "age" (возраст жилья)
        // Предполагаем наличие колонок year_of_sale и year_built
        Dataset<Row> dfWithAge = df.withColumn("age",
                when(col("year_of_sale").isNotNull().and(col("year_built").isNotNull()),
                        col("year_of_sale").minus(col("year_built")))
                        .otherwise(lit(null)));

        log.info("Column 'age' added");
        dfWithAge.select("year_of_sale", "year_built", "age").show(10, false);

        // Задание 2: Средняя дата продажи для сочетаний zip_code и tax_class
        // Используем year_of_sale как proxy для даты продажи
        Dataset<Row> avgSaleDateByZipTax = dfWithAge.groupBy("zip_code", "tax_class")
                .agg(avg("year_of_sale").alias("avg_sale_year"))
                .orderBy("zip_code", "tax_class");

        log.info("Average sale year by zip_code and tax_class");
        avgSaleDateByZipTax.show(20, false);

        // Задание 3: Суммарная стоимость жилья по сочетаниям tax_class и zip_code
        Dataset<Row> totalPriceByTaxZip = dfWithAge.groupBy("tax_class", "zip_code")
                .agg(sum("sale_price").alias("total_sale_price"))
                .orderBy("tax_class", "zip_code");

        log.info("Total sale price by tax_class and zip_code");
        totalPriceByTaxZip.show(20, false);

        // Задание 4: Выбрать 10 колонок с null, но не преобладающими, удалить строки с полными null
        // Получаем общее количество строк
        long totalRows = dfWithAge.count();
        log.info("Total rows: {}", totalRows);

        // Собираем имена колонок, где доля null > 0 и < 0.5
        String[] allColumns = dfWithAge.columns();
        List<String> selectedColumns = new ArrayList<>();

        for (String colName : allColumns) {
            long nullCount = dfWithAge.filter(col(colName).isNull()).count();
            double nullRatio = (double) nullCount / totalRows;
            if (nullRatio > 0 && nullRatio < 0.5) {
                selectedColumns.add(colName);
                log.debug("Column {} has null ratio {}", colName, nullRatio);
            }
        }

        // Если колонок больше 10, берем первые 10
        if (selectedColumns.size() > 10) {
            selectedColumns = selectedColumns.subList(0, 10);
        }

        log.info("Selected {} columns for further filtering: {}", selectedColumns.size(), selectedColumns);

        // Создаем новый DataFrame только с выбранными колонками
        // Используем select с переменным числом аргументов
        String[] selectedColumnsArray = selectedColumns.toArray(new String[0]);
        Dataset<Row> dfSelected = dfWithAge.select(selectedColumnsArray[0],
                java.util.Arrays.copyOfRange(selectedColumnsArray, 1, selectedColumnsArray.length));

        // Удаляем строки, где все выбранные колонки равны null
        // Строим условие: не (колонка1 is null AND колонка2 is null AND ...)
        Column allNullCondition = null;
        for (String colName : selectedColumns) {
            Column isNullCond = col(colName).isNull();
            if (allNullCondition == null) {
                allNullCondition = isNullCond;
            } else {
                allNullCondition = allNullCondition.and(isNullCond);
            }
        }
        // Оставляем строки, где не все колонки null
        Dataset<Row> dfFiltered = dfSelected.filter(not(allNullCondition));

        log.info("Rows before filtering: {}, after filtering: {}", dfSelected.count(), dfFiltered.count());
        dfFiltered.show(20, false);

        // Завершение работы
        spark.stop();
        log.info("Spark session stopped");
    }
}