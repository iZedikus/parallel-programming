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
public class MulticoreImageProcessor {
    private static final int[][] EMBOSS_KERNEL = {
            {-2, -1, 0},
            {-1,  1, 1},
            { 0,  1, 2}
    };
    private static final int KERNEL_SIZE = 3;
    private static final int DIVISOR = 1;
    private static final int OFFSET = 128;
    private static final int BLOCK_SIZE = 32;

    public static void main(String[] args) {
        if (args.length != 2) {
            log.error("Использование: java MultiThreadedImageReliefProcessor <input.jpg> <output.jpg>");
            return;
        }

        for (int i = 2; i <= 16; i += 2) {
            String[] newArgs = new String[]{args[0], args[1], String.valueOf(i)};
            processImage(newArgs);
        }
    }

    private static void processImage(String[] args) {
        String inputPath = args[0];
        String outputPath = args[1];
        int numThreads = Integer.parseInt(args[2]);

        try {
            BufferedImage original = ImageIO.read(new File(inputPath));
            if (original == null) {
                log.error("Не удалось загрузить изображение: " + inputPath);
                return;
            }

            Instant beginning = Instant.now();

            BufferedImage relieved = processReliefMultiThreaded(original, numThreads);
            BufferedImage downscaled = processDownscaleMultiThreaded(relieved, numThreads);

            ImageIO.write(downscaled, "jpg", new File(outputPath));

            Duration duration = Duration.between(beginning, Instant.now());

            log.info("Обработка завершена за {} с {} потоками.", duration, numThreads);
        } catch (IOException | InterruptedException | ExecutionException e) {
            System.err.println("Ошибка обработки: " + e.getMessage());
        }
    }

    private static BufferedImage processReliefMultiThreaded(BufferedImage image, int numThreads)
            throws InterruptedException, ExecutionException {
        int width = image.getWidth();
        int height = image.getHeight();
        BufferedImage result = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<Future<?>> futures = new ArrayList<>();

        int numBlocksX = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int numBlocksY = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int blockY = 0; blockY < numBlocksY; blockY++) {
            for (int blockX = 0; blockX < numBlocksX; blockX++) {
                final int startX = blockX * BLOCK_SIZE;
                final int startY = blockY * BLOCK_SIZE;
                final int endX = Math.min(startX + BLOCK_SIZE, width);
                final int endY = Math.min(startY + BLOCK_SIZE, height);

                Future<?> future = executor.submit(() ->
                        applyReliefBlock(image, result, startX, startY, endX, endY));
                futures.add(future);
            }
        }

        for (Future<?> future : futures) {
            future.get();
        }
        executor.shutdown();
        if (executor.awaitTermination(10, TimeUnit.SECONDS)) {
            throw new RuntimeException("Прошло 10 секунд. Прекращение работы...");
        }

        return result;
    }

    private static void applyReliefBlock(BufferedImage src, BufferedImage dst,
                                         int startX, int startY, int endX, int endY) {
        for (int y = startY; y < endY; y++) {
            for (int x = startX; x < endX; x++) {
                int[] components = convolute(src, x, y);
                int rgb = (Math.max(0, Math.min(255, components[0])) << 16) |
                        (Math.max(0, Math.min(255, components[1])) << 8) |
                        Math.max(0, Math.min(255, components[2]));
                dst.setRGB(x, y, rgb);
            }
        }
    }

    private static int[] convolute(BufferedImage image, int cx, int cy) {
        int halfKernel = KERNEL_SIZE / 2;
        int[] sum = new int[3]; // R, G, B

        for (int ky = -halfKernel; ky <= halfKernel; ky++) {
            for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                int x = cx + kx;
                int y = cy + ky;
                int kValue = MulticoreImageProcessor.EMBOSS_KERNEL[ky + halfKernel][kx + halfKernel];

                int rgb = getPixelRGB(image, x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                sum[0] += kValue * r;
                sum[1] += kValue * g;
                sum[2] += kValue * b;
            }
        }

        for (int i = 0; i < 3; i++) {
            sum[i] = (sum[i] / DIVISOR) + OFFSET;
        }
        return sum;
    }

    private static int getPixelRGB(BufferedImage image, int x, int y) {
        int width = image.getWidth();
        int height = image.getHeight();

        if (x < 0 || x >= width || y < 0 || y >= height) {
            return 128 << 16 | 128 << 8 | 128;
        }
        return image.getRGB(x, y);
    }

    private static BufferedImage processDownscaleMultiThreaded(BufferedImage image, int numThreads)
            throws InterruptedException, ExecutionException {
        int origWidth = image.getWidth();
        int origHeight = image.getHeight();
        int newWidth = origWidth / 2;
        int newHeight = origHeight / 2;
        BufferedImage result = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB);

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<Future<?>> futures = new ArrayList<>();

        int numBlocksX = (newWidth + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int numBlocksY = (newHeight + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int blockY = 0; blockY < numBlocksY; blockY++) {
            for (int blockX = 0; blockX < numBlocksX; blockX++) {
                final int startX = blockX * BLOCK_SIZE;
                final int startY = blockY * BLOCK_SIZE;
                final int endX = Math.min(startX + BLOCK_SIZE, newWidth);
                final int endY = Math.min(startY + BLOCK_SIZE, newHeight);

                Future<?> future = executor.submit(() ->
                        downscaleBlock(image, result, startX, startY, endX, endY, origWidth, origHeight));
                futures.add(future);
            }
        }

        for (Future<?> future : futures) {
            future.get();
        }
        executor.shutdown();
        if (executor.awaitTermination(10, TimeUnit.SECONDS)) {
            throw new RuntimeException("Прошло 10 секунд. Прекращение работы...");
        }

        return result;
    }

    private static void downscaleBlock(BufferedImage src, BufferedImage dst,
                                       int startX, int startY, int endX, int endY,
                                       int origWidth, int origHeight) {
        for (int y = startY; y < endY; y++) {
            for (int x = startX; x < endX; x++) {
                int sumR = 0, sumG = 0, sumB = 0;
                int count = 0;

                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int origX = x * 2 + dx;
                        int origY = y * 2 + dy;
                        if (origX < origWidth && origY < origHeight) {
                            int rgb = src.getRGB(origX, origY);
                            sumR += (rgb >> 16) & 0xFF;
                            sumG += (rgb >> 8) & 0xFF;
                            sumB += rgb & 0xFF;
                            count++;
                        }
                    }
                }

                int avgR = sumR / count;
                int avgG = sumG / count;
                int avgB = sumB / count;

                int rgb = (avgR << 16) | (avgG << 8) | avgB;
                dst.setRGB(x, y, rgb);
            }
        }
    }
}
