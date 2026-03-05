package ru.stepanov;

import lombok.extern.slf4j.Slf4j;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.*;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.*;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;

import static org.apache.spark.sql.functions.*;

@Slf4j
public class FetalHealthML {

    public static void main(String[] args) {
        // 1. Создание SparkSession
        SparkSession spark = SparkSession.builder()
                .appName("FetalHealthML")
                .master("local[*]")
                .getOrCreate();

        log.info("Spark session created");

        // 2. Загрузка данных (путь к файлу можно изменить или передать аргументом)
        String filePath = "/Users/avrellian/Documents/Java Projects/parallel_programming/lab5/src/main/resources/fetal_health.csv";
        Dataset<Row> df = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(filePath);

        log.info("Data loaded from {}", filePath);
        df.printSchema();
        df.show(5, false);

        // 3. Подготовка данных: преобразование метки в бинарную (Double)
        // Отрицательный класс: 1 (Normal) -> 0.0
        // Положительный класс: 2 (Suspect) и 3 (Pathological) -> 1.0
        Dataset<Row> preparedDf = df.withColumn("label",
                when(col("fetal_health").equalTo(1), lit(0.0))
                        .otherwise(lit(1.0)));

        // Удалим исходную колонку fetal_health, оставим только признаки и новую метку
        String[] featureColumns = df.columns(); // все колонки исходного df
        // Убираем 'fetal_health' из списка признаков
        String[] featureCols = java.util.Arrays.stream(featureColumns)
                .filter(c -> !c.equals("fetal_health"))
                .toArray(String[]::new);

        log.info("Feature columns: {}", (Object) featureCols);

        // 4. Создание VectorAssembler для объединения признаков
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("features");

        // 5. Разделение на train/test (80% обучение, 20% тест)
        Dataset<Row>[] splits = preparedDf.randomSplit(new double[]{0.8, 0.2}, 42);
        Dataset<Row> train = splits[0];
        Dataset<Row> test = splits[1];

        log.info("Train set size: {}, Test set size: {}", train.count(), test.count());

        // 6. Метрика для кросс-валидации (используем площадь под ROC)
        BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator()
                .setMetricName("areaUnderROC")
                .setLabelCol("label")
                .setRawPredictionCol("rawPrediction");

        // ========================= Модель 1: Логистическая регрессия =========================
        log.info("=== Training Logistic Regression ===");
        LogisticRegression lr = new LogisticRegression()
                .setLabelCol("label")
                .setFeaturesCol("features");

        // Сетка параметров для логистической регрессии
        ParamMap[] lrParamGrid = new ParamGridBuilder()
                .addGrid(lr.maxIter(), new int[]{10, 100, 1000, 10000})
                .addGrid(lr.regParam(), new double[]{0.1, 0.5, 1.0, 2.0})
                .addGrid(lr.elasticNetParam(), new double[]{0.0, 0.5, 1.0})
                .build();

        // Конвейер для LR
        Pipeline lrPipeline = new Pipeline().setStages(new PipelineStage[]{assembler, lr});

        // Кросс-валидация с 3 фолдами
        CrossValidator lrCV = new CrossValidator()
                .setEstimator(lrPipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(lrParamGrid)
                .setNumFolds(3);

        CrossValidatorModel lrModel = lrCV.fit(train);
        // Получаем лучшую модель (PipelineModel), из неё извлекаем стадию классификатора
        PipelineModel bestPipelineModel = (PipelineModel) lrModel.bestModel();
        LogisticRegressionModel bestLrModel = (LogisticRegressionModel) bestPipelineModel.stages()[1];

        log.info("Best LogisticRegression params: maxIter={}, regParam={}, elasticNetParam={}",
                bestLrModel.getMaxIter(), bestLrModel.getRegParam(), bestLrModel.getElasticNetParam());

        // Предсказание на тесте
        Dataset<Row> lrPredictions = bestPipelineModel.transform(test);
        printMetrics(lrPredictions, "Logistic Regression");

        // ========================= Модель 2: Дерево решений =========================
        log.info("=== Training Decision Tree ===");
        DecisionTreeClassifier dt = new DecisionTreeClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setImpurity("gini");

        ParamMap[] dtParamGrid = new ParamGridBuilder()
                .addGrid(dt.maxDepth(), new int[]{3, 5, 9, 12})
                .build();

        Pipeline dtPipeline = new Pipeline().setStages(new PipelineStage[]{assembler, dt});
        CrossValidator dtCV = new CrossValidator()
                .setEstimator(dtPipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(dtParamGrid)
                .setNumFolds(3);

        CrossValidatorModel dtModel = dtCV.fit(train);
        bestPipelineModel = (PipelineModel) dtModel.bestModel();
        DecisionTreeClassificationModel bestDtModel = (DecisionTreeClassificationModel) bestPipelineModel.stages()[1];

        log.info("Best DecisionTree params: maxDepth={}", bestDtModel.getMaxDepth());

        Dataset<Row> dtPredictions = bestPipelineModel.transform(test);
        printMetrics(dtPredictions, "Decision Tree");

        // ========================= Модель 3: Случайный лес =========================
        log.info("=== Training Random Forest ===");
        RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setImpurity("gini");

        ParamMap[] rfParamGrid = new ParamGridBuilder()
                .addGrid(rf.maxDepth(), new int[]{3, 5, 9, 12})
                .addGrid(rf.numTrees(), new int[]{5, 11, 25})
                .build();

        Pipeline rfPipeline = new Pipeline().setStages(new PipelineStage[]{assembler, rf});
        CrossValidator rfCV = new CrossValidator()
                .setEstimator(rfPipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(rfParamGrid)
                .setNumFolds(3);

        CrossValidatorModel rfModel = rfCV.fit(train);
        bestPipelineModel = (PipelineModel) rfModel.bestModel();
        RandomForestClassificationModel bestRfModel = (RandomForestClassificationModel) bestPipelineModel.stages()[1];

        log.info("Best RandomForest params: maxDepth={}, numTrees={}",
                bestRfModel.getMaxDepth(), bestRfModel.getNumTrees());

        Dataset<Row> rfPredictions = bestPipelineModel.transform(test);
        printMetrics(rfPredictions, "Random Forest");

        // 10. Завершение
        spark.stop();
        log.info("Spark session stopped");
    }

    /**
     * Вычисляет и выводит метрики: Confusion Matrix, Accuracy, Precision, Recall
     * @param predictions DataFrame с колонками "prediction" и "label"
     * @param modelName название модели для логирования
     */
    private static void printMetrics(Dataset<Row> predictions, String modelName) {
        // Преобразуем колонки prediction и label в Double на случай, если они не Double
        Dataset<Row> metricsDf = predictions.select(
                col("prediction").cast("double"),
                col("label").cast("double")
        );

        // Для использования MulticlassMetrics нужно преобразовать в RDD
        MulticlassMetrics metrics = new MulticlassMetrics(
                metricsDf.toJavaRDD()
                        .map(row -> new scala.Tuple2<>(
                                row.getDouble(0),  // prediction
                                row.getDouble(1)   // label
                        ))
                        .rdd()
        );

        // Матрица ошибок
        org.apache.spark.mllib.linalg.Matrix confusion = metrics.confusionMatrix();
        log.info("{} - Confusion Matrix:\n{}", modelName, confusion.toString());

        // Accuracy
        double accuracy = metrics.accuracy();
        log.info("{} - Accuracy: {}", modelName, accuracy);

        // Precision и Recall для положительного класса (label = 1)
        double precision = metrics.precision(1);
        double recall = metrics.recall(1);
        log.info("{} - Precision (positive class): {}", modelName, precision);
        log.info("{} - Recall (positive class): {}", modelName, recall);

        // Дополнительно: F1-мера
        double f1 = metrics.fMeasure(1);
        log.info("{} - F1 (positive class): {}", modelName, f1);
    }
}