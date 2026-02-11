package ru.stepanov;

import lombok.extern.slf4j.Slf4j;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;

import java.awt.image.BufferedImage;

import static org.jocl.CL.CL_CONTEXT_DEVICES;
import static org.jocl.CL.CL_CONTEXT_PLATFORM;
import static org.jocl.CL.CL_DEVICE_AVAILABLE;
import static org.jocl.CL.CL_DEVICE_GLOBAL_MEM_SIZE;
import static org.jocl.CL.CL_DEVICE_LOCAL_MEM_SIZE;
import static org.jocl.CL.CL_DEVICE_MAX_COMPUTE_UNITS;
import static org.jocl.CL.CL_DEVICE_MAX_WORK_GROUP_SIZE;
import static org.jocl.CL.CL_DEVICE_MAX_WORK_ITEM_SIZES;
import static org.jocl.CL.CL_DEVICE_NAME;
import static org.jocl.CL.CL_DEVICE_OPENCL_C_VERSION;
import static org.jocl.CL.CL_DEVICE_TYPE;
import static org.jocl.CL.CL_DEVICE_TYPE_ACCELERATOR;
import static org.jocl.CL.CL_DEVICE_TYPE_ALL;
import static org.jocl.CL.CL_DEVICE_TYPE_CPU;
import static org.jocl.CL.CL_DEVICE_TYPE_GPU;
import static org.jocl.CL.CL_DEVICE_VENDOR;
import static org.jocl.CL.CL_DEVICE_VERSION;
import static org.jocl.CL.CL_MEM_COPY_HOST_PTR;
import static org.jocl.CL.CL_MEM_READ_ONLY;
import static org.jocl.CL.CL_MEM_WRITE_ONLY;
import static org.jocl.CL.CL_PLATFORM_NAME;
import static org.jocl.CL.CL_PLATFORM_VENDOR;
import static org.jocl.CL.CL_PLATFORM_VERSION;
import static org.jocl.CL.CL_PROGRAM_BUILD_LOG;
import static org.jocl.CL.CL_SUCCESS;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clBuildProgram;
import static org.jocl.CL.clCreateBuffer;
import static org.jocl.CL.clCreateCommandQueue;
import static org.jocl.CL.clCreateContextFromType;
import static org.jocl.CL.clCreateKernel;
import static org.jocl.CL.clCreateProgramWithSource;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clEnqueueReadBuffer;
import static org.jocl.CL.clFinish;
import static org.jocl.CL.clGetContextInfo;
import static org.jocl.CL.clGetDeviceIDs;
import static org.jocl.CL.clGetDeviceInfo;
import static org.jocl.CL.clGetPlatformIDs;
import static org.jocl.CL.clGetPlatformInfo;
import static org.jocl.CL.clGetProgramBuildInfo;
import static org.jocl.CL.clReleaseCommandQueue;
import static org.jocl.CL.clReleaseContext;
import static org.jocl.CL.clReleaseKernel;
import static org.jocl.CL.clReleaseMemObject;
import static org.jocl.CL.clReleaseProgram;
import static org.jocl.CL.clSetKernelArg;

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

            log.debug("Найдено платформ OpenCL: {}", platforms.length);

            for (int i = 0; i < platforms.length; i++) {
                cl_platform_id platform = platforms[i];

                // Получаем информацию о платформе
                long[] size = new long[1];

                // Имя платформы
                clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, null, size);
                byte[] nameBytes = new byte[(int)size[0]];
                clGetPlatformInfo(platform, CL_PLATFORM_NAME, nameBytes.length, Pointer.to(nameBytes), null);
                String platformName = new String(nameBytes, 0, nameBytes.length - 1);

                // Вендор платформы
                clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, null, size);
                byte[] vendorBytes = new byte[(int)size[0]];
                clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorBytes.length, Pointer.to(vendorBytes), null);
                String platformVendor = new String(vendorBytes, 0, vendorBytes.length - 1);

                // Версия платформы
                clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, null, size);
                byte[] versionBytes = new byte[(int)size[0]];
                clGetPlatformInfo(platform, CL_PLATFORM_VERSION, versionBytes.length, Pointer.to(versionBytes), null);
                String platformVersion = new String(versionBytes, 0, versionBytes.length - 1);

                log.debug("Платформа {}: {}", i, platformName);
                log.debug("  Вендор: {}", platformVendor);
                log.debug("  Версия: {}", platformVersion);

                // Получаем устройства для этой платформы
                int[] numDevices = new int[1];
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, null, numDevices);

                if (numDevices[0] > 0) {
                    cl_device_id[] devices = new cl_device_id[numDevices[0]];
                    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devices.length, devices, null);

                    for (int j = 0; j < devices.length; j++) {
                        cl_device_id device = devices[j];

                        // Тип устройства
                        long[] deviceType = new long[1];
                        clGetDeviceInfo(device, CL_DEVICE_TYPE, Sizeof.cl_long, Pointer.to(deviceType), null);
                        String typeStr;
                        if ((deviceType[0] & CL_DEVICE_TYPE_GPU) != 0) {
                            typeStr = "GPU";
                        } else if ((deviceType[0] & CL_DEVICE_TYPE_CPU) != 0) {
                            typeStr = "CPU";
                        } else if ((deviceType[0] & CL_DEVICE_TYPE_ACCELERATOR) != 0) {
                            typeStr = "ACCELERATOR";
                        } else {
                            typeStr = "OTHER";
                        }

                        // Имя устройства
                        clGetDeviceInfo(device, CL_DEVICE_NAME, 0, null, size);
                        byte[] deviceNameBytes = new byte[(int)size[0]];
                        clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameBytes.length, Pointer.to(deviceNameBytes), null);
                        String deviceName = new String(deviceNameBytes, 0, deviceNameBytes.length - 1);

                        // Вендор устройства
                        clGetDeviceInfo(device, CL_DEVICE_VENDOR, 0, null, size);
                        byte[] deviceVendorBytes = new byte[(int)size[0]];
                        clGetDeviceInfo(device, CL_DEVICE_VENDOR, deviceVendorBytes.length, Pointer.to(deviceVendorBytes), null);
                        String deviceVendor = new String(deviceVendorBytes, 0, deviceVendorBytes.length - 1);

                        // Версия драйвера
                        clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, null, size);
                        byte[] driverVersionBytes = new byte[(int)size[0]];
                        clGetDeviceInfo(device, CL_DEVICE_VERSION, driverVersionBytes.length, Pointer.to(driverVersionBytes), null);
                        String driverVersion = new String(driverVersionBytes, 0, driverVersionBytes.length - 1);

                        // Максимальная глобальная память
                        long[] globalMemSize = new long[1];
                        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, Sizeof.cl_long, Pointer.to(globalMemSize), null);

                        // Максимальная локальная память
                        long[] localMemSize = new long[1];
                        clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, Sizeof.cl_long, Pointer.to(localMemSize), null);

                        // Максимальный размер рабочей группы
                        long[] maxWorkGroupSize = new long[1];
                        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, Sizeof.cl_long, Pointer.to(maxWorkGroupSize), null);

                        // Максимальные размеры глобальной рабочей области
                        long[] maxWorkItemSizes = new long[3];
                        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, Sizeof.cl_long * 3, Pointer.to(maxWorkItemSizes), null);

                        // Количество вычислительных блоков
                        long[] maxComputeUnits = new long[1];
                        clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, Sizeof.cl_long, Pointer.to(maxComputeUnits), null);

                        // Поддержка OpenCL C
                        byte[] openclCVersionBytes = new byte[128];
                        clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, openclCVersionBytes.length, Pointer.to(openclCVersionBytes), null);
                        String openclCVersion = new String(openclCVersionBytes).trim();

                        log.debug("""
                                        Устройство {}: {} ({})
                                          Вендор: {}
                                          Версия: {}
                                          OpenCL C версия: {}
                                          Глобальная память: {} MB
                                          Локальная память: {} KB
                                          Макс. размер рабочей группы: {}
                                          Макс. размеры рабочей области: [{}, {}, {}]
                                          Кол-во вычислительных блоков: {}""",
                                j, deviceName, typeStr,
                                deviceVendor,
                                driverVersion,
                                openclCVersion,
                                globalMemSize[0] / (1024 * 1024),
                                localMemSize[0] / 1024,
                                maxWorkGroupSize[0],
                                maxWorkItemSizes[0], maxWorkItemSizes[1], maxWorkItemSizes[2],
                                maxComputeUnits[0]);

                        // Проверка доступности
                        long[] available = new long[1];
                        clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, Sizeof.cl_long, Pointer.to(available), null);
                        log.debug("      Доступно: {}", available[0] != 0);
                    }
                }
            }

            // Выбираем первую платформу с GPU
            cl_platform_id platform = platforms[0];
            boolean gpuFound = false;

            for (cl_platform_id plat : platforms) {
                int[] numDevices = new int[1];
                clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, null, numDevices);

                if (numDevices[0] > 0) {
                    platform = plat;
                    gpuFound = true;
                    break;
                }
            }

            if (!gpuFound) {
                log.warn("GPU не найдено, используем CPU устройство");
            }

            // Создаем контекст
            cl_context_properties contextProperties = new cl_context_properties();
            contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

            // Инвертируем, потому что я не могу найти GPU с достаточным количеством памяти
            if (!gpuFound) {
                context = clCreateContextFromType(contextProperties,
                        CL_DEVICE_TYPE_GPU, null, null, null);
            } else {
                context = clCreateContextFromType(contextProperties,
                        CL_DEVICE_TYPE_CPU, null, null, null);
            }

            // Получаем устройство из контекста
            cl_device_id[] devices = new cl_device_id[1];
            clGetContextInfo(context, CL_CONTEXT_DEVICES,
                    Sizeof.cl_device_id, Pointer.to(devices), null);

            // Получаем информацию о выбранном устройстве
            long[] size = new long[1];

            // Имя выбранного устройства
            clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, null, size);
            byte[] deviceNameBytes = new byte[(int)size[0]];
            clGetDeviceInfo(devices[0], CL_DEVICE_NAME, deviceNameBytes.length, Pointer.to(deviceNameBytes), null);
            String deviceName = new String(deviceNameBytes, 0, deviceNameBytes.length - 1);

            log.debug("Выбрано устройство: {}", deviceName);

            // Создаем очередь команд
            commandQueue = clCreateCommandQueue(context, devices[0], 0, null);

            // Создаем и компилируем программу
            program = clCreateProgramWithSource(context, 1,
                    new String[]{CONVOLUTION_KERNEL_SOURCE}, null, null);

            // Компилируем с проверкой ошибок
            int buildError = clBuildProgram(program, 0, null, null, null, null);
            if (buildError != CL_SUCCESS) {
                // Получаем сообщение об ошибке
                byte[] buffer = new byte[10240];
                clGetProgramBuildInfo(program, devices[0],
                        CL_PROGRAM_BUILD_LOG, buffer.length, Pointer.to(buffer), null);
                String buildLog = new String(buffer);
                log.error("Ошибка компиляции OpenCL программы:");
                log.error("{}", buildLog.trim());
                throw new RuntimeException("OpenCL program build failed: " + buildError);
            }

            // Создаем ядра
            convolutionKernel = clCreateKernel(program, "convolution", null);
            if (convolutionKernel == null) {
                throw new RuntimeException("Не удалось создать ядро convolution");
            }

            downscaleKernel = clCreateKernel(program, "downscale", null);
            if (downscaleKernel == null) {
                throw new RuntimeException("Не удалось создать ядро downscale");
            }

            log.info("OpenCL успешно инициализирован");
            log.info("Используется устройство: {}", deviceName);

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
                (long) Sizeof.cl_float * totalSize,
                Pointer.to(input), null);

        cl_mem outputBuffer = clCreateBuffer(context,
                CL_MEM_WRITE_ONLY,
                (long) Sizeof.cl_float * totalSize,
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
                (long) Sizeof.cl_float * input.length,
                Pointer.to(input), null);

        cl_mem outputBuffer = clCreateBuffer(context,
                CL_MEM_WRITE_ONLY,
                (long) Sizeof.cl_float * newTotalSize,
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
}