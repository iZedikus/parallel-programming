package ru.stepanov;

import lombok.extern.slf4j.Slf4j;
import org.jocl.*;

import java.awt.image.BufferedImage;

import static org.jocl.CL.*;

@Slf4j
public class GPUImageProcessor {

    private cl_context context;
    private cl_command_queue commandQueue;
    private cl_program program;
    private cl_kernel convolutionKernel;
    private cl_kernel downscaleKernel;

    // Ядро для рельефа (emboss filter)
    private static final float[] EMBOSS_KERNEL = {
            -2.0f, -1.0f,  0.0f,
            -1.0f,  1.0f,  1.0f,
            0.0f,  1.0f,  2.0f
    };

    // Исходный код OpenCL ядер
    private static final String CONVOLUTION_KERNEL_SOURCE =
            """
                    __kernel void convolution(
                        __global const float* input,
                        __global float* output,
                        const int width,
                        const int height,
                        __constant float* kernel,
                        const int channelOffset,
                        const int totalChannels)
                    {
                        int x = get_global_id(0);
                        int y = get_global_id(1);
                       \s
                        if (x >= width || y >= height) return;
                       \s
                        float sum = 0.0f;
                        int kernelRadius = 1;
                       \s
                        for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
                            for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
                                int px = x + kx;
                                int py = y + ky;
                               \s
                                if (px >= 0 && px < width && py >= 0 && py < height) {
                                    int kernelIndex = (ky + kernelRadius) * 3 + (kx + kernelRadius);
                                    int pixelIndex = (py * width + px) * totalChannels + channelOffset;
                                    sum += input[pixelIndex] * kernel[kernelIndex];
                                }
                            }
                        }
                       \s
                        int outputIndex = (y * width + x) * totalChannels + channelOffset;
                        output[outputIndex] = clamp(sum, 0.0f, 1.0f);
                    }
                    
                    __kernel void downscale(
                        __global const float* input,
                        __global float* output,
                        const int width,
                        const int height,
                        const int newWidth,
                        const int newHeight,
                        const int channelOffset,
                        const int totalChannels)
                    {
                        int x = get_global_id(0);
                        int y = get_global_id(1);
                       \s
                        if (x >= newWidth || y >= newHeight) return;
                       \s
                        float sum = 0.0f;
                       \s
                        for (int dy = 0; dy < 2; dy++) {
                            for (int dx = 0; dx < 2; dx++) {
                                int srcX = x * 2 + dx;
                                int srcY = y * 2 + dy;
                               \s
                                if (srcX < width && srcY < height) {
                                    int srcIndex = (srcY * width + srcX) * totalChannels + channelOffset;
                                    sum += input[srcIndex];
                                }
                            }
                        }
                       \s
                        int dstIndex = (y * newWidth + x) * totalChannels + channelOffset;
                        output[dstIndex] = sum / 4.0f;
                    }""";

    public GPUImageProcessor() {
        initOpenCL();
    }

    private void initOpenCL() {
        try {
            // Инициализация платформы
            cl_platform_id platform = getPlatform();

            // Инициализация контекста
            cl_context_properties contextProperties = new cl_context_properties();
            contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

            context = clCreateContextFromType(contextProperties,
                    CL_DEVICE_TYPE_GPU, null, null, null);

            // Получение устройства
            cl_device_id[] devices = new cl_device_id[1];
            clGetContextInfo(context, CL_CONTEXT_DEVICES,
                    Sizeof.cl_device_id, Pointer.to(devices), null);

            // Создание очереди команд
            commandQueue = clCreateCommandQueue(context, devices[0], 0, null);

            // Создание программы
            program = clCreateProgramWithSource(context, 1,
                    new String[]{CONVOLUTION_KERNEL_SOURCE}, null, null);

            // Компиляция программы
            clBuildProgram(program, 0, null, null, null, null);

            // Создание ядер
            convolutionKernel = clCreateKernel(program, "convolution", null);
            downscaleKernel = clCreateKernel(program, "downscale", null);

            log.info("OpenCL инициализирован успешно");

        } catch (Exception e) {
            log.error("Ошибка инициализации OpenCL", e);
            throw new RuntimeException("Не удалось инициализировать OpenCL", e);
        }
    }

    private cl_platform_id getPlatform() {
        int[] numPlatforms = new int[1];
        clGetPlatformIDs(0, null, numPlatforms);

        cl_platform_id[] platforms = new cl_platform_id[numPlatforms[0]];
        clGetPlatformIDs(platforms.length, platforms, null);

        return platforms[0];
    }

    public BufferedImage processImage(BufferedImage inputImage) {
        long startTime = System.currentTimeMillis();

        int width = inputImage.getWidth();
        int height = inputImage.getHeight();
        int channels = 3; // RGB

        // Преобразование изображения в массив float (0-1)
        float[] imageData = imageToFloatArray(inputImage);

        // Применение свертки для каждого канала
        float[] convolvedData = applyConvolution(imageData, width, height, channels);

        // Уменьшение масштаба
        float[] downscaledData = applyDownscale(convolvedData, width, height, channels);

        int newWidth = width / 2;
        int newHeight = height / 2;

        // Преобразование обратно в изображение
        BufferedImage outputImage = floatArrayToImage(downscaledData, newWidth, newHeight);

        long endTime = System.currentTimeMillis();
        log.info("Обработка изображения {}x{} заняла {} мс",
                width, height, (endTime - startTime));

        return outputImage;
    }

    private float[] imageToFloatArray(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        float[] data = new float[width * height * 3];

        int index = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);

                // Извлечение каналов и нормализация к [0, 1]
                data[index++] = ((rgb >> 16) & 0xFF) / 255.0f; // R
                data[index++] = ((rgb >> 8) & 0xFF) / 255.0f;  // G
                data[index++] = (rgb & 0xFF) / 255.0f;         // B
            }
        }

        return data;
    }

    private float[] applyConvolution(float[] input, int width, int height, int channels) {
        int totalPixels = width * height;
        int totalSize = totalPixels * channels;

        // Создание буферов в памяти GPU
        cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * totalSize, Pointer.to(input), null);
        cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * totalSize, null, null);
        cl_mem kernelBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * 9, Pointer.to(EMBOSS_KERNEL), null);

        float[] output = new float[totalSize];

        try {
            // Установка аргументов ядра свертки
            clSetKernelArg(convolutionKernel, 0, Sizeof.cl_mem, Pointer.to(inputBuffer));
            clSetKernelArg(convolutionKernel, 1, Sizeof.cl_mem, Pointer.to(outputBuffer));
            clSetKernelArg(convolutionKernel, 2, Sizeof.cl_int, Pointer.to(new int[]{width}));
            clSetKernelArg(convolutionKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{height}));
            clSetKernelArg(convolutionKernel, 4, Sizeof.cl_mem, Pointer.to(kernelBuffer));

            // Обработка каждого канала
            for (int channel = 0; channel < channels; channel++) {
                clSetKernelArg(convolutionKernel, 5, Sizeof.cl_int, Pointer.to(new int[]{channel}));
                clSetKernelArg(convolutionKernel, 6, Sizeof.cl_int, Pointer.to(new int[]{channels}));

                // Запуск ядра
                long[] globalWorkSize = new long[]{width, height};
                clEnqueueNDRangeKernel(commandQueue, convolutionKernel, 2, null,
                        globalWorkSize, null, 0, null, null);
            }

            // Чтение результата
            clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0,
                    Sizeof.cl_float * totalSize, Pointer.to(output), 0, null, null);

        } finally {
            clReleaseMemObject(inputBuffer);
            clReleaseMemObject(outputBuffer);
            clReleaseMemObject(kernelBuffer);
        }

        return output;
    }

    private float[] applyDownscale(float[] input, int width, int height, int channels) {
        int newWidth = width / 2;
        int newHeight = height / 2;
        int newTotalSize = newWidth * newHeight * channels;

        // Создание буферов
        cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * input.length, Pointer.to(input), null);
        cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * newTotalSize, null, null);

        float[] output = new float[newTotalSize];

        try {
            // Установка аргументов ядра уменьшения
            clSetKernelArg(downscaleKernel, 0, Sizeof.cl_mem, Pointer.to(inputBuffer));
            clSetKernelArg(downscaleKernel, 1, Sizeof.cl_mem, Pointer.to(outputBuffer));
            clSetKernelArg(downscaleKernel, 2, Sizeof.cl_int, Pointer.to(new int[]{width}));
            clSetKernelArg(downscaleKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{height}));
            clSetKernelArg(downscaleKernel, 4, Sizeof.cl_int, Pointer.to(new int[]{newWidth}));
            clSetKernelArg(downscaleKernel, 5, Sizeof.cl_int, Pointer.to(new int[]{newHeight}));

            // Обработка каждого канала
            for (int channel = 0; channel < channels; channel++) {
                clSetKernelArg(downscaleKernel, 6, Sizeof.cl_int, Pointer.to(new int[]{channel}));
                clSetKernelArg(downscaleKernel, 7, Sizeof.cl_int, Pointer.to(new int[]{channels}));

                // Запуск ядра
                long[] globalWorkSize = new long[]{newWidth, newHeight};
                clEnqueueNDRangeKernel(commandQueue, downscaleKernel, 2, null,
                        globalWorkSize, null, 0, null, null);
            }

            // Чтение результата
            clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0,
                    Sizeof.cl_float * newTotalSize, Pointer.to(output), 0, null, null);

        } finally {
            clReleaseMemObject(inputBuffer);
            clReleaseMemObject(outputBuffer);
        }

        return output;
    }

    private BufferedImage floatArrayToImage(float[] data, int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        int index = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int r = (int) (data[index++] * 255);
                int g = (int) (data[index++] * 255);
                int b = (int) (data[index++] * 255);

                // Ограничение значений
                r = Math.min(255, Math.max(0, r));
                g = Math.min(255, Math.max(0, g));
                b = Math.min(255, Math.max(0, b));

                int rgb = (r << 16) | (g << 8) | b;
                image.setRGB(x, y, rgb);
            }
        }

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
}