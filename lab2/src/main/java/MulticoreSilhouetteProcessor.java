import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReaderBuilder;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.stat.descriptive.rank.Min;
import org.apache.commons.math3.stat.descriptive.rank.Max;
import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;

import java.io.FileReader;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

@Slf4j
public class MulticoreSilhouetteProcessor {
    private static final int BLOCK_SIZE = 64;  // Размер блока для многопоточной обработки
    private static final int TIMEOUT_SEC = 60; // Таймаут ожидания завершения потоков
    private static final double EPSILON = 1e-10; // Маленькая константа для избежания деления на ноль

    public static void main(String[] args) {
        if (args.length != 1) {
            log.error("Использование: java MulticoreSilhouetteCalculator <BD-Patients.csv>");
            return;
        }

        for (int clusterSize = 1000; clusterSize <= 5000; clusterSize += 2000) {
            try {
                RealMatrix data = loadAndNormalizeData(args[0], clusterSize);
                for (int k = 3; k <= 5; k++) {
                    log.info("Запущена обработка {} векторов в {} кластерах", clusterSize, k);
                    List<CentroidCluster<DoublePoint>> clusters = proceedClusterization(data, k);

                    for (int i = 0; i < k; i++) {
                        double[] center = clusters.get(i).getCenter().getPoint();
                        log.debug("Центр кластера {}: [{}, {}]", i, center[0], center[1]);
                    }

                    for (int numThreads = 2; numThreads <= 16; numThreads += 2) {
                        Instant beginning = Instant.now();

                        double silhouetteScore = calculateSilhouette(data, clusters, numThreads);

                        Duration duration = Duration.between(beginning, Instant.now());
                        log.info("{} потоков завершили обработку за {} миллисекунд. Силуэт: {}",
                                numThreads,
                                duration.toMillis(),
                                silhouetteScore
                        );
                    }
                    System.out.println();
                }
                System.out.println();
            } catch (IOException e) {
                log.error("Ошибка при парсинге данных");
            } catch (Exception e) {
                log.error("Ошибка при вычислении силуэта: {}", e.getMessage());
            }
        }
    }

    private static RealMatrix loadAndNormalizeData(String csvPath, int sampleSize) throws IOException {
        List<double[]> dataPoints = new ArrayList<>();

        try (CSVReader csvReader = new CSVReaderBuilder(new FileReader(csvPath))
                .withCSVParser(new CSVParserBuilder()
                        .withSeparator(';')
                        .build())
                .build()) {

            csvReader.readNext(); // Пропускаем заголовок

            String[] line;
            int processed = 0;

            while ((line = csvReader.readNext()) != null && processed < sampleSize) {
                if (line.length >= 2) {
                    try {
                        double creat = Double.parseDouble(line[0].trim());
                        double hco3 = Double.parseDouble(line[1].trim());
                        dataPoints.add(new double[]{creat, hco3});
                        processed++;
                    } catch (NumberFormatException e) {
                        log.warn("Пропущена некорректная строка: {}", String.join(",", line));
                    }
                }
            }
        } catch (CsvValidationException e) {
            throw new IOException("Ошибка парсинга CSV: " + e.getMessage(), e);
        }

        if (dataPoints.isEmpty()) {
            throw new IOException("Не удалось загрузить данные из " + csvPath);
        }
        log.debug("{} строк из файла успешно обработаны", dataPoints.size());

        // Нормализация в диапазон [0,1] для каждого признака
        double[] creatMinMax = findMinMax(dataPoints, 0);
        double[] hco3MinMax = findMinMax(dataPoints, 1);

        log.debug("Диапазон Creatinine: [{}, {}]", creatMinMax[0], creatMinMax[1]);
        log.debug("Диапазон HCO3_mean: [{}, {}]", hco3MinMax[0], hco3MinMax[1]);

        RealMatrix normalized = new Array2DRowRealMatrix(dataPoints.size(), 2);
        for (int i = 0; i < dataPoints.size(); i++) {
            double creatNorm = (dataPoints.get(i)[0] - creatMinMax[0]) / (creatMinMax[1] - creatMinMax[0] + EPSILON);
            double hco3Norm = (dataPoints.get(i)[1] - hco3MinMax[0]) / (hco3MinMax[1] - hco3MinMax[0] + EPSILON);
            normalized.setRow(i, new double[]{Math.max(0, Math.min(1, creatNorm)),
                    Math.max(0, Math.min(1, hco3Norm))});
        }

        return normalized;
    }

    private static List<CentroidCluster<DoublePoint>> proceedClusterization(RealMatrix data, int k) {
        List<DoublePoint> points = new ArrayList<>();
        for (int i = 0; i < data.getRowDimension(); i++) {
            points.add(new DoublePoint(data.getRow(i)));
        }

        KMeansPlusPlusClusterer<DoublePoint> clusterer = new KMeansPlusPlusClusterer<>(
                k, 100, new EuclideanDistance());

        return clusterer.cluster(points);
    }

    private static double[] findMinMax(List<double[]> data, int featureIndex) {
        Min minFunc = new Min();
        Max maxFunc = new Max();
        double[] values = data.stream().mapToDouble(point -> point[featureIndex]).toArray();
        return new double[]{minFunc.evaluate(values), maxFunc.evaluate(values)};
    }

    private static double calculateSilhouette(RealMatrix data,
                                              List<CentroidCluster<DoublePoint>> clusters,
                                              int numThreads)
            throws InterruptedException, ExecutionException {

        int n = data.getRowDimension();
        double[] silhouetteValues = new double[n];

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<Future<?>> futures = new ArrayList<>();

        int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int block = 0; block < numBlocks; block++) {
            final int startIdx = block * BLOCK_SIZE;
            final int endIdx = Math.min(startIdx + BLOCK_SIZE, n);

            Future<?> future = executor.submit(() ->
                    computeSilhouetteBlock(data.getData(), clusters, silhouetteValues, startIdx, endIdx));
            futures.add(future);
        }

        for (Future<?> future : futures) {
            future.get();
        }

        executor.shutdown();
        if (!executor.awaitTermination(TIMEOUT_SEC, TimeUnit.SECONDS)) {
            throw new RuntimeException("Таймаут превышен");
        }

        return average(silhouetteValues);
    }

    private static void computeSilhouetteBlock(double[][] points,
                                               List<CentroidCluster<DoublePoint>> clusters,
                                               double[] silhouetteValues,
                                               int startIdx, int endIdx) {
        for (int i = startIdx; i < endIdx; i++) {
            double[] point = points[i];

            // Находим кластер точки
            int clusterId = findClusterMembership(point, clusters);
            List<DoublePoint> sameClusterPoints = clusters.get(clusterId).getPoints();

            // a(i) - среднее расстояние до точек своего кластера
            double a = averageDistanceToPoints(point, sameClusterPoints);

            // b(i) - минимальное среднее расстояние до других кластеров
            double minB = Double.MAX_VALUE;
            for (int c = 0; c < clusters.size(); c++) {
                if (c != clusterId) {
                    double b = averageDistanceToPoints(point, clusters.get(c).getPoints());
                    if (b < minB) minB = b;
                }
            }

            silhouetteValues[i] = (minB - a) / Math.max(Math.max(a, minB), EPSILON);
        }
    }

    private static int findClusterMembership(double[] point, List<CentroidCluster<DoublePoint>> clusters) {
        EuclideanDistance distance = new EuclideanDistance();
        double minDist = Double.MAX_VALUE;
        int bestCluster = 0;

        for (int i = 0; i < clusters.size(); i++) {
            double[] center = clusters.get(i).getCenter().getPoint();
            double dist = distance.compute(point, center);
            if (dist < minDist) {
                minDist = dist;
                bestCluster = i;
            }
        }
        return bestCluster;
    }

    private static double averageDistanceToPoints(double[] point, List<DoublePoint> clusterPoints) {
        EuclideanDistance distance = new EuclideanDistance();
        double sum = 0;
        int count = clusterPoints.size();

        if (count == 0) return 0;

        for (DoublePoint p : clusterPoints) {
            sum += distance.compute(point, p.getPoint());
        }
        return sum / count;
    }

    private static double average(double[] values) {
        double sum = 0;
        for (double v : values) {
            sum += v;
        }
        return sum / values.length;
    }
}
