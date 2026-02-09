import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;

public class ImageReliefProcessor {
    private static final int[][] EMBOSS_KERNEL = {
            {-2, -1, 0},
            {-1,  1, 1},
            { 0,  1, 2}
    };
    private static final int KERNEL_SIZE = 3;
    private static final int DIVISOR = 1;
    private static final int OFFSET = 128;

    public static void main(String[] args) {
        if (args.length != 2) {
            System.err.println("Использование: java ImageReliefProcessor <input.jpg> <output.jpg>");
            return;
        }

        String inputPath = args[0];
        String outputPath = args[1];

        try {
            BufferedImage original = ImageIO.read(new File(inputPath));
            if (original == null) {
                System.err.println("Не удалось загрузить изображение: " + inputPath);
                return;
            }

            Instant beginning = Instant.now();

            BufferedImage relieved = applyRelief(original);
            BufferedImage downscaled = downscaleBy2(relieved);

            Duration workingTime = Duration.between(beginning, Instant.now());

            ImageIO.write(downscaled, "jpg", new File(outputPath));
            System.out.printf("""
                    Результат сохранен: %s,
                    Время выполнения обработки: %d милисекунд
                    """, outputPath, workingTime.toMillis());
        } catch (IOException e) {
            System.err.println("Ошибка обработки: " + e.getMessage());
        }
    }

    private static BufferedImage applyRelief(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        BufferedImage result = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int[] components = convolute(image, x, y);
                int rgb = (Math.max(0, Math.min(255, components[0])) << 16) |
                        (Math.max(0, Math.min(255, components[1])) << 8) |
                        Math.max(0, Math.min(255, components[2]));
                result.setRGB(x, y, rgb);
            }
        }
        return result;
    }

    private static int[] convolute(BufferedImage image, int cx, int cy) {
        int halfKernel = KERNEL_SIZE / 2;
        int[] sum = new int[3]; // R, G, B

        for (int ky = -halfKernel; ky <= halfKernel; ky++) {
            for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                int x = cx + kx;
                int y = cy + ky;
                int kValue = EMBOSS_KERNEL[ky + halfKernel][kx + halfKernel];

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
            return 128 << 16 | 128 << 8 | 128; // Серый для границ (экстраполяция)
        }
        return image.getRGB(x, y);
    }

    private static BufferedImage downscaleBy2(BufferedImage image) {
        int origWidth = image.getWidth();
        int origHeight = image.getHeight();
        int newWidth = origWidth / 2;
        int newHeight = origHeight / 2;

        BufferedImage result = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < newHeight; y++) {
            for (int x = 0; x < newWidth; x++) {
                int sumR = 0, sumG = 0, sumB = 0;
                int count = 0;

                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int origX = x * 2 + dx;
                        int origY = y * 2 + dy;
                        if (origX < origWidth && origY < origHeight) {
                            int rgb = image.getRGB(origX, origY);
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
                result.setRGB(x, y, rgb);
            }
        }
        return result;
    }
}
