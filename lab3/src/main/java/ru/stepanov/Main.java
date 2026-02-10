package ru.stepanov;

import lombok.extern.slf4j.Slf4j;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;

@Slf4j
public class Main {

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Использование:");
            System.out.println("java -jar gpu-image-processing.jar <входной файл> <выходной файл>");
            return;
        }

        processSingleImage(args[0], args[1]);
    }

    private static void processSingleImage(String inputPath, String outputPath) {
        try {
            log.info("Загрузка изображения: {}", inputPath);

            File inputFile = new File(inputPath);
            BufferedImage inputImage = ImageIO.read(inputFile);

            log.info("Размер изображения: {}x{}",
                    inputImage.getWidth(), inputImage.getHeight());

            GPUImageProcessor processor = new GPUImageProcessor();
            BufferedImage outputImage = processor.processImage(inputImage);

            File outputFile = new File(outputPath);
            ImageIO.write(outputImage, "jpg", outputFile);

            log.info("Изображение сохранено: {}", outputPath);

            processor.cleanup();

        } catch (Exception e) {
            log.error("Ошибка обработки изображения", e);
        }
    }
}