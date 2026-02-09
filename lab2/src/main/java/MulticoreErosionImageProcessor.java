import lombok.extern.slf4j.Slf4j;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

@Slf4j
public class MulticoreErosionImageProcessor {
    private static final int THRESHOLD = 150;
    private static final int EROSION_STEP = 2;
    private static final int BLOCK_SIZE = 1024;
    private static final int TIMEOUT_SEC = 30;

    public static void main(String[] args) {
        if (args.length != 2) {
            log.error("Использование: java MulticoreErosionImageProcessor <input.jpg> <output.jpg>");
            return;
        }

        for (int i = 2; i <= 16; i += 2) {
            String[] newArgs = new String[]{args[0], args[1], String.valueOf(i)};
            processImage(newArgs);
        }
    }

    private static void processImage(String[] args) {
        log.info("Запущена обработка файла {} на {} потоках", args[0], args[2]);
        String inputPath = args[0];
        String outputPath = args[1];
        int numThreads = Integer.parseInt(args[2]);

        try {
            BufferedImage original = ImageIO.read(new File(inputPath));
            if (original == null) {
                log.error("Не удалось загрузить изображение: {}", inputPath);
                return;
            }

            Instant beginning = Instant.now();

            BufferedImage processed = processBinarizationAndErosion(original, numThreads);

            Duration duration = Duration.between(beginning, Instant.now());

            ImageIO.write(processed, "jpg", new File(outputPath));

            log.info("{} потока(ов) завершили обработку за {} миллисекунд.", numThreads, duration.toMillis());
        } catch (IOException | InterruptedException | ExecutionException e) {
            log.error("Ошибка обработки: {}", e.getMessage());
        }
    }

    private static BufferedImage processBinarizationAndErosion(BufferedImage image, int numThreads)
            throws InterruptedException, ExecutionException {
        int width = image.getWidth();
        int height = image.getHeight();

        BufferedImage binarized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        applyBinarizationMultiThreaded(image, binarized, numThreads);

        BufferedImage eroded = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        applyErosionMultiThreaded(binarized, eroded, numThreads);

        return eroded;
    }

    private static void applyBinarizationMultiThreaded(BufferedImage src, BufferedImage dst, int numThreads)
            throws InterruptedException, ExecutionException {
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<Future<?>> futures = new ArrayList<>();

        int numBlocksX = (src.getWidth() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int numBlocksY = (src.getHeight() + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int blockY = 0; blockY < numBlocksY; blockY++) {
            for (int blockX = 0; blockX < numBlocksX; blockX++) {
                final int startX = blockX * BLOCK_SIZE;
                final int startY = blockY * BLOCK_SIZE;
                final int endX = Math.min(startX + BLOCK_SIZE, src.getWidth());
                final int endY = Math.min(startY + BLOCK_SIZE, src.getHeight());

                Future<?> future = executor.submit(() ->
                        binarizeBlock(src, dst, startX, startY, endX, endY));
                futures.add(future);
            }
        }

        for (Future<?> future : futures) future.get();
        executor.shutdown();
        if (!executor.awaitTermination(TIMEOUT_SEC, TimeUnit.SECONDS)) {
            throw new RuntimeException("Прошло " + TIMEOUT_SEC + " секунд. Прекращение работы...");
        }
    }

    private static void binarizeBlock(BufferedImage src, BufferedImage dst,
                                      int startX, int startY, int endX, int endY) {
        for (int y = startY; y < endY; y++) {
            for (int x = startX; x < endX; x++) {
                int intensity = getPixelIntensity(src, x, y);
                int value = (intensity < THRESHOLD) ? 0 : 255; // 0=черный, 255=белый
                dst.setRGB(x, y, (value << 16) | (value << 8) | value);
            }
        }
    }

    private static void applyErosionMultiThreaded(BufferedImage src, BufferedImage dst, int numThreads)
            throws InterruptedException, ExecutionException {
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<Future<?>> futures = new ArrayList<>();

        int numBlocksX = (src.getWidth() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int numBlocksY = (src.getHeight() + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int blockY = 0; blockY < numBlocksY; blockY++) {
            for (int blockX = 0; blockX < numBlocksX; blockX++) {
                final int startX = blockX * BLOCK_SIZE;
                final int startY = blockY * BLOCK_SIZE;
                final int endX = Math.min(startX + BLOCK_SIZE, src.getWidth());
                final int endY = Math.min(startY + BLOCK_SIZE, src.getHeight());

                Future<?> future = executor.submit(() ->
                        erodeBlock(src, dst, startX, startY, endX, endY));
                futures.add(future);
            }
        }

        for (Future<?> future : futures) future.get();
        executor.shutdown();
        if (!executor.awaitTermination(TIMEOUT_SEC, TimeUnit.SECONDS)) {
            throw new RuntimeException("Прошло " + TIMEOUT_SEC + " секунд. Прекращение работы...");
        }
    }

    private static void erodeBlock(BufferedImage src, BufferedImage dst,
                                   int startX, int startY, int endX, int endY) {
        for (int y = startY; y < endY; y++) {
            for (int x = startX; x < endX; x++) {
                boolean allWhite = true;
                for (int dy = -EROSION_STEP; dy <= EROSION_STEP && allWhite; dy++) {
                    for (int dx = -EROSION_STEP; dx <= EROSION_STEP && allWhite; dx++) {
                        int nx = x + dx;
                        int ny = y + dy;
                        if (!isPixelWhite(src, nx, ny)) {
                            allWhite = false;
                        }
                    }
                }
                int value = allWhite ? 255 : 0;
                dst.setRGB(x, y, (value << 16) | (value << 8) | value);
            }
        }
    }

    private static int getPixelIntensity(BufferedImage image, int x, int y) {
        int rgb = getPixelRGB(image, x, y);
        int r = (rgb >> 16) & 0xFF;
        int g = (rgb >> 8) & 0xFF;
        int b = rgb & 0xFF;
        return (r + g + b) / 3;
    }

    private static int getPixelRGB(BufferedImage image, int x, int y) {
        int width = image.getWidth();
        int height = image.getHeight();
        if (x < 0 || x >= width || y < 0 || y >= height) {
            return 128 << 16 | 128 << 8 | 128;
        }
        return image.getRGB(x, y);
    }

    private static boolean isPixelWhite(BufferedImage image, int x, int y) {
        int rgb = getPixelRGB(image, x, y);
        int r = (rgb >> 16) & 0xFF;
        int g = (rgb >> 8) & 0xFF;
        int b = rgb & 0xFF;
        return r > 200 && g > 200 && b > 200;
    }
}
