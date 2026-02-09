import com.opencsv.CSVParser;
import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import com.opencsv.exceptions.CsvValidationException;
import lombok.extern.slf4j.Slf4j;
import lombok.Data;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@Slf4j
public class MulticoreKMeansClustering {

    @Data
    static class Cluster {
        double[] centroid;
        List<Integer> points = new ArrayList<>();
    }

    public static void main(String[] args) {
        if (args.length != 1) {
            log.error("Использование: java KMeansClustering <path>");
            return;
        }

        try {
            double[][] data = loadCsv(args[0]);
            double[][] scaledData = scaleToZeroOne(data);

            log.info("Загружено {} строк, выполнена кластеризация для K=3,4,5", scaledData.length);

            for (int k = 3; k <= 5; k++) {
                performClustering(scaledData, k);
            }

            benchmarkSilhouette(scaledData);

        } catch (Exception e) {
            log.error("Ошибка обработки", e);
        }
    }

    static double[][] loadCsv(String path) throws IOException, CsvValidationException {
        try (CSVReader reader = new CSVReaderBuilder(new FileReader(path))
                .withCSVParser(new CSVParserBuilder()
                        .withSeparator(';')
                        .build()
                ).build()) {
            reader.readNext(); // пропускаем заголовок
            List<double[]> data = new ArrayList<>();

            String[] line;
            while ((line = reader.readNext()) != null) {
                data.add(new double[]{Double.parseDouble(line[0]), Double.parseDouble(line[1])});
            }

            return data.toArray(new double[0][]);
        }
    }

    static double[][] scaleToZeroOne(double[][] data) {
        int n = data.length, d = data[0].length;
        double[][] scaled = new double[n][d];

        for (int feature = 0; feature < d; feature++) {
            final int effectivelyFinalFeature = feature;

            double min = Arrays.stream(data).mapToDouble(row -> row[effectivelyFinalFeature]).min().orElse(0);
            double max = Arrays.stream(data).mapToDouble(row -> row[effectivelyFinalFeature]).max().orElse(1);
            double range = max - min;
            if (range == 0) range = 1;

            for (int i = 0; i < n; i++) {
                scaled[i][feature] = (data[i][feature] - min) / range;
            }
        }
        return scaled;
    }

    static void performClustering(double[][] data, int k) {
        int[] labels = kMeans(data, k);
        double avgSilhouette = calculateSilhouette(data, labels, Runtime.getRuntime().availableProcessors());

        log.info("K={}, средний силуэт={}", k, avgSilhouette);
        logCentroids(data, labels, k);
    }

    static int[] kMeans(double[][] data, int k) {
        int n = data.length;
        Random rand = new Random(42);
        Cluster[] clusters = initClusters(data, k, rand);

        int[] labels = new int[n];
        boolean changed;

        do {
            changed = assignPoints(data, clusters, labels);
            updateCentroids(data, clusters);
        } while (changed);

        return labels;
    }

    static Cluster[] initClusters(double[][] data, int k, Random rand) {
        Cluster[] clusters = new Cluster[k];
        for (int i = 0; i < k; i++) {
            clusters[i] = new Cluster();
            clusters[i].centroid = data[rand.nextInt(data.length)].clone();
        }
        return clusters;
    }

    static boolean assignPoints(double[][] data, Cluster[] clusters, int[] labels) {
        boolean changed = false;
        for (Cluster c : clusters) c.points.clear();

        for (int i = 0; i < data.length; i++) {
            double minDist = Double.MAX_VALUE;
            int bestCluster = 0;

            for (int j = 0; j < clusters.length; j++) {
                double dist = euclideanDistance(data[i], clusters[j].centroid);
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = j;
                }
            }

            if (labels[i] != bestCluster) changed = true;
            labels[i] = bestCluster;
            clusters[bestCluster].points.add(i);
        }
        return changed;
    }

    static void updateCentroids(double[][] data, Cluster[] clusters) {
        for (Cluster cluster : clusters) {
            if (cluster.points.isEmpty()) continue;

            double[] newCentroid = new double[data[0].length];
            for (int idx : cluster.points) {
                for (int d = 0; d < newCentroid.length; d++) {
                    newCentroid[d] += data[idx][d];
                }
            }

            int size = cluster.points.size();
            for (int d = 0; d < newCentroid.length; d++) {
                newCentroid[d] /= size;
            }
            cluster.centroid = newCentroid;
        }
    }

    static void logCentroids(double[][] data, int[] labels, int k) {
        Map<Integer, List<Integer>> clusterPoints = new HashMap<>();
        for (int i = 0; i < labels.length; i++) {
            clusterPoints.computeIfAbsent(labels[i], x -> new ArrayList<>()).add(i);
        }

        log.info("  Центроиды:");
        for (int c = 0; c < k; c++) {
            List<Integer> points = clusterPoints.getOrDefault(c, Collections.emptyList());
            if (!points.isEmpty()) {
                double[] centroid = new double[2];
                for (int idx : points) {
                    centroid[0] += data[idx][0];
                    centroid[1] += data[idx][1];
                }
                centroid[0] /= points.size();
                centroid[1] /= points.size();
                log.info("Кластер {}: ({}, {}), размер={}",
                        c, centroid[0], centroid[1], points.size());
            }
        }
    }

    static double calculateSilhouette(double[][] data, int[] labels, int numThreads) {
        double[] silhouettes = new double[data.length];
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        List<Future<?>> futures = IntStream.range(0, data.length)
                .mapToObj(i -> executor.submit(() -> {
                    silhouettes[i] = silhouetteScore(data, labels, i);
                }))
                .collect(Collectors.toList());

        futures.forEach(future -> {
            try {
                future.get();
            } catch (Exception e) {
                log.error("Ошибка вычисления силуэта", e);
            }
        });
        executor.shutdown();

        return Arrays.stream(silhouettes).average().orElse(0);
    }

    static double silhouetteScore(double[][] data, int[] labels, int i) {
        int cluster = labels[i];
        List<Integer> sameCluster = getClusterPoints(labels, cluster);
        double a = avgDistance(data, i, sameCluster);

        double b = Double.MAX_VALUE;
        for (int c = 0; c < Arrays.stream(labels).max().orElse(0) + 1; c++) {
            if (c == cluster) continue;
            List<Integer> otherCluster = getClusterPoints(labels, c);
            double dist = avgDistance(data, i, otherCluster);
            if (dist < b) b = dist;
        }

        return (b - a) / Math.max(a, b);
    }

    static List<Integer> getClusterPoints(int[] labels, int cluster) {
        List<Integer> points = new ArrayList<>();
        for (int i = 0; i < labels.length; i++) {
            if (labels[i] == cluster) points.add(i);
        }
        return points;
    }

    static double avgDistance(double[][] data, int i, List<Integer> points) {
        if (points.size() <= 1) return 0;
        double sum = 0;
        for (int j : points) {
            if (i != j) sum += euclideanDistance(data[i], data[j]);
        }
        return sum / (points.size() - 1);
    }

    static double euclideanDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    static void benchmarkSilhouette(double[][] data) {
        int[] sizes = {1000, 3000, 5000};
        int[] threads = {2, 4, 6, 8, 10, 12, 14, 16};

        int[] labels = kMeans(data, 4); // фиксируем K=4

        log.info("Бенчмарк индекса силуэта:");
        for (int size : sizes) {
            if (size > data.length) continue;

            double[][] subset = Arrays.copyOfRange(data, 0, size);
            int[] subsetLabels = Arrays.copyOfRange(labels, 0, size);

            log.info("Размер={}:", size);
            for (int t : threads) {
                long start = System.nanoTime();
                double sil = calculateSilhouette(subset, subsetLabels, t);
                long time = (System.nanoTime() - start) / 1_000_000;
                log.info("{} потоков: силуэт={}, время={}мс", t, sil, time);
            }
        }
    }
}
