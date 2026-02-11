package ru.stepanov;

import lombok.extern.slf4j.Slf4j;
import org.jocl.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;

import static org.jocl.CL.*;

@Slf4j
public class GPUImageProcessor {

    private cl_context context;
    private cl_command_queue commandQueue;
    private cl_program program;
    private cl_kernel convolutionKernel;
    private cl_kernel downscaleKernel;

    private static final float[] EMBOSS_KERNEL = {
            -2.0f, -1.0f,  0.0f,
            -1.0f,  1.0f,  1.0f,
            0.0f,  1.0f,  2.0f
    };

    private static final String CONVOLUTION_KERNEL_SOURCE =
            """
                    __kernel void convolution(
                        __global const float* input,
                        __global float* output,
                        const int width,
                        const int height,
                        __constant float* kernel1)
                    {
                        int x = get_global_id(0);
                        int y = get_global_id(1);
                       \s
                        if (x >= width || y >= height) return;
                       \s
                        int kernelRadius = 1;
                        float3 sum = (float3)(0.0f, 0.0f, 0.0f);
                        float kernelSum = 0.0f;
                       \s
                        for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
                            for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
                                int px = x + kx;
                                int py = y + ky;
                               \s
                                if (px >= 0 && px < width && py >= 0 && py < height) {
                                    int kernelIndex = (ky + kernelRadius) * 3 + (kx + kernelRadius);
                                    float kernelValue = kernel1[kernelIndex];
                                   \s
                                    int pixelIndex = (py * width + px) * 3;
                                   \s
                                    float r = input[pixelIndex];
                                    float g = input[pixelIndex + 1];
                                    float b = input[pixelIndex + 2];
                                   \s
                                    sum.x += r * kernelValue;
                                    sum.y += g * kernelValue;
                                    sum.z += b * kernelValue;
                                    kernelSum += kernelValue;
                                }
                            }
                        }
                       \s
                        int outputIndex = (y * width + x) * 3;
                       \s
                        if (kernelSum != 0.0f) {
                            output[outputIndex] = clamp(sum.x / kernelSum, 0.0f, 1.0f);
                            output[outputIndex + 1] = clamp(sum.y / kernelSum, 0.0f, 1.0f);
                            output[outputIndex + 2] = clamp(sum.z / kernelSum, 0.0f, 1.0f);
                        } else {
                            output[outputIndex] = input[outputIndex];
                            output[outputIndex + 1] = input[outputIndex + 1];
                            output[outputIndex + 2] = input[outputIndex + 2];
                        }
                    }
                    
                    __kernel void downscale(
                        __global const float* input,
                        __global float* output,
                        const int width,
                        const int height,
                        const int newWidth,
                        const int newHeight)
                    {
                        int x = get_global_id(0);
                        int y = get_global_id(1);
                       \s
                        if (x >= newWidth || y >= newHeight) return;
                       \s
                        float3 sum = (float3)(0.0f, 0.0f, 0.0f);
                        int count = 0;
                       \s
                        for (int dy = 0; dy < 2; dy++) {
                            for (int dx = 0; dx < 2; dx++) {
                                int srcX = x * 2 + dx;
                                int srcY = y * 2 + dy;
                               \s
                                if (srcX < width && srcY < height) {
                                    int srcIndex = (srcY * width + srcX) * 3;
                                   \s
                                    sum.x += input[srcIndex];
                                    sum.y += input[srcIndex + 1];
                                    sum.z += input[srcIndex + 2];
                                    count++;
                                }
                            }
                        }
                       \s
                        int dstIndex = (y * newWidth + x) * 3;
                       \s
                        if (count > 0) {
                            output[dstIndex] = sum.x / count;
                            output[dstIndex + 1] = sum.y / count;
                            output[dstIndex + 2] = sum.z / count;
                        }
                    }""";

    public GPUImageProcessor() {
        initOpenCL();
    }

    private void initOpenCL() {
        try {
            // Получаем доступные платформы
            int[] numPlatforms = new int[1];
            clGetPlatformIDs(0, null, numPlatforms);

            cl_platform_id[] platforms = new cl_platform_id[numPlatforms[0]];
            clGetPlatformIDs(platforms.length, platforms, null);

            cl_platform_id platform = platforms[0];

            // Создаем контекст
            cl_context_properties contextProperties = new cl_context_properties();
            contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

            context = clCreateContextFromType(contextProperties,
                    CL_DEVICE_TYPE_GPU, null, null, null);

            // Получаем устройство
            cl_device_id[] devices = new cl_device_id[1];
            clGetContextInfo(context, CL_CONTEXT_DEVICES,
                    Sizeof.cl_device_id, Pointer.to(devices), null);

            if (devices[0] == null) {
                throw new RuntimeException("GPU device not found");
            }

            // Создаем очередь команд
            commandQueue = clCreateCommandQueue(context, devices[0], 0, null);

            // Создаем и компилируем программу
            program = clCreateProgramWithSource(context, 1,
                    new String[]{CONVOLUTION_KERNEL_SOURCE}, null, null);

            // Компилируем с проверкой ошибок
            int buildError = clBuildProgram(program, 0, null, null, null, null);
            if (buildError != CL_SUCCESS) {
                // Получаем сообщение об ошибке
                byte[] buffer = new byte[1024];
                clGetProgramBuildInfo(program, devices[0],
                        CL_PROGRAM_BUILD_LOG, buffer.length, Pointer.to(buffer), null);
                String buildLog = new String(buffer);
                log.error("Build log: {}", buildLog);
                throw new RuntimeException("OpenCL program build failed: " + buildError);
            }

            // Создаем ядра
            convolutionKernel = clCreateKernel(program, "convolution", null);
            downscaleKernel = clCreateKernel(program, "downscale", null);

            log.info("OpenCL успешно инициализирован");

        } catch (Exception e) {
            log.error("Ошибка инициализации OpenCL", e);
            throw new RuntimeException("Не удалось инициализировать OpenCL", e);
        }
    }

    public BufferedImage processImage(BufferedImage inputImage) {
        long startTime = System.currentTimeMillis();

        int width = inputImage.getWidth();
        int height = inputImage.getHeight();

        // Преобразуем изображение в массив float (0-1)
        log.debug("Конвертация изображения в float массив");
        float[] imageData = imageToFloatArray(inputImage);

        // Применяем свертку
        log.debug("Применение свертки");
        float[] convolvedData = applyConvolution(imageData, width, height);

        // Уменьшаем масштаб
        log.debug("Уменьшение масштаба");
        float[] downscaledData = applyDownscale(convolvedData, width, height);

        int newWidth = width / 2;
        int newHeight = height / 2;

        // Преобразуем обратно в изображение
        log.debug("Конвертация результата в изображение");
        BufferedImage outputImage = floatArrayToImage(downscaledData, newWidth, newHeight);

        long endTime = System.currentTimeMillis();
        log.info("Обработка изображения {}x{} заняла {} мс",
                width, height, (endTime - startTime));

        return outputImage;
    }

    private float[] applyConvolution(float[] input, int width, int height) {
        int channels = 3;
        int totalPixels = width * height;
        int totalSize = totalPixels * channels;

        log.debug("Создание OpenCL буферов для свертки");

        // Создаем буферы
        cl_mem inputBuffer = clCreateBuffer(context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * totalSize,
                Pointer.to(input), null);

        cl_mem outputBuffer = clCreateBuffer(context,
                CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * totalSize,
                null, null);

        cl_mem kernelBuffer = clCreateBuffer(context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * 9,
                Pointer.to(EMBOSS_KERNEL), null);

        float[] output = new float[totalSize];

        try {
            // Устанавливаем аргументы ядра
            clSetKernelArg(convolutionKernel, 0, Sizeof.cl_mem, Pointer.to(inputBuffer));
            clSetKernelArg(convolutionKernel, 1, Sizeof.cl_mem, Pointer.to(outputBuffer));
            clSetKernelArg(convolutionKernel, 2, Sizeof.cl_int, Pointer.to(new int[]{width}));
            clSetKernelArg(convolutionKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{height}));
            clSetKernelArg(convolutionKernel, 4, Sizeof.cl_mem, Pointer.to(kernelBuffer));

            // Запускаем ядро
            long[] globalWorkSize = new long[]{width, height};
            clEnqueueNDRangeKernel(commandQueue, convolutionKernel, 2, null,
                    globalWorkSize, null, 0, null, null);

            // Синхронизируем
            clFinish(commandQueue);

            // Читаем результат
            clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0,
                    Sizeof.cl_float * totalSize, Pointer.to(output), 0, null, null);

            log.debug("Свертка завершена. Первые 6 значений: {}, {}, {}, {}, {}, {}",
                    output[0], output[1], output[2], output[3], output[4], output[5]);

        } finally {
            // Освобождаем ресурсы
            clReleaseMemObject(inputBuffer);
            clReleaseMemObject(outputBuffer);
            clReleaseMemObject(kernelBuffer);
        }

        return output;
    }

    private float[] applyDownscale(float[] input, int width, int height) {
        int channels = 3;
        int newWidth = width / 2;
        int newHeight = height / 2;
        int newTotalSize = newWidth * newHeight * channels;

        log.debug("Создание OpenCL буферов для уменьшения масштаба");

        // Создаем буферы
        cl_mem inputBuffer = clCreateBuffer(context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * input.length,
                Pointer.to(input), null);

        cl_mem outputBuffer = clCreateBuffer(context,
                CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * newTotalSize,
                null, null);

        float[] output = new float[newTotalSize];

        try {
            // Устанавливаем аргументы ядра
            clSetKernelArg(downscaleKernel, 0, Sizeof.cl_mem, Pointer.to(inputBuffer));
            clSetKernelArg(downscaleKernel, 1, Sizeof.cl_mem, Pointer.to(outputBuffer));
            clSetKernelArg(downscaleKernel, 2, Sizeof.cl_int, Pointer.to(new int[]{width}));
            clSetKernelArg(downscaleKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{height}));
            clSetKernelArg(downscaleKernel, 4, Sizeof.cl_int, Pointer.to(new int[]{newWidth}));
            clSetKernelArg(downscaleKernel, 5, Sizeof.cl_int, Pointer.to(new int[]{newHeight}));

            // Запускаем ядро
            long[] globalWorkSize = new long[]{newWidth, newHeight};
            clEnqueueNDRangeKernel(commandQueue, downscaleKernel, 2, null,
                    globalWorkSize, null, 0, null, null);

            // Синхронизируем
            clFinish(commandQueue);

            // Читаем результат
            clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0,
                    Sizeof.cl_float * newTotalSize, Pointer.to(output), 0, null, null);

        } finally {
            clReleaseMemObject(inputBuffer);
            clReleaseMemObject(outputBuffer);
        }

        return output;
    }

    private float[] imageToFloatArray(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        float[] data = new float[width * height * 3];

        int[] rgbArray = image.getRGB(0, 0, width, height, null, 0, width);

        for (int i = 0; i < rgbArray.length; i++) {
            int rgb = rgbArray[i];

            data[i * 3] = ((rgb >> 16) & 0xFF) / 255.0f;     // R
            data[i * 3 + 1] = ((rgb >> 8) & 0xFF) / 255.0f;  // G
            data[i * 3 + 2] = (rgb & 0xFF) / 255.0f;         // B
        }

        return data;
    }

    private BufferedImage floatArrayToImage(float[] data, int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        int[] rgbArray = new int[width * height];

        for (int i = 0; i < rgbArray.length; i++) {
            int r = (int) (data[i * 3] * 255);
            int g = (int) (data[i * 3 + 1] * 255);
            int b = (int) (data[i * 3 + 2] * 255);

            // Ограничиваем значения
            r = Math.max(0, Math.min(255, r));
            g = Math.max(0, Math.min(255, g));
            b = Math.max(0, Math.min(255, b));

            rgbArray[i] = (r << 16) | (g << 8) | b;
        }

        image.setRGB(0, 0, width, height, rgbArray, 0, width);
        return image;
    }

    public void cleanup() {
        if (convolutionKernel != null) clReleaseKernel(convolutionKernel);
        if (downscaleKernel != null) clReleaseKernel(downscaleKernel);
        if (program != null) clReleaseProgram(program);
        if (commandQueue != null) clReleaseCommandQueue(commandQueue);
        if (context != null) clReleaseContext(context);

        log.info("Ресурсы OpenCL освобождены");
    }

    // Метод для тестирования
    public static void testSimple() {
        try {
            // Создаем простое тестовое изображение 4x4
            BufferedImage testImage = new BufferedImage(4, 4, BufferedImage.TYPE_INT_RGB);

            // Заполняем изображение
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    int rgb = (x * 63) << 16 | (y * 63) << 8 | ((x + y) * 31);
                    testImage.setRGB(x, y, rgb);
                }
            }

            GPUImageProcessor processor = new GPUImageProcessor();
            BufferedImage result = processor.processImage(testImage);

            // Сохраняем результат
            ImageIO.write(result, "png", new File("test_result.png"));
            System.out.println("Тест завершен успешно. Результат сохранен в test_result.png");

            processor.cleanup();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        // Простой тест
        testSimple();
    }
}