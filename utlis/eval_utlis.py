import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import correlate

def calculate_mse_loss(generated_images, real_images):
    """
    计算生成图像与真实图像之间的均方误差MSE损失。
    
    :param generated_images: 由CGAN生成的图像张量 形状为(N, C, H, W)
    :param real_images: 对应的真实图像张量，形状为(N, C, H, W)
    :return: MSE损失的平均值
    """
    mse_loss = F.mse_loss(generated_images, real_images, reduction='mean')
    return mse_loss

def ssim(img1, img2, window_size=11, size_average=True):
    """
    计算两幅图像之间的结构相似度指数(SSIM)。
    
    :param img1: 第一幅图像张量
    :param img2: 第二幅图像张量
    :param window_size: SSIM计算中使用的高斯核大小
    :param size_average: 是否对所有batch求平均值
    :return: SSIM值
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = F.avg_pool2d(img1, window_size, 1, window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, 1, window_size // 2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1.pow(2), window_size, 1, window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2.pow(2), window_size, 1, window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, window_size // 2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)




def compute_time_autocorrelation(data, max_lag=None):
    """
    计算数据的时间维度自相关。
    
    :param data: 形状为(batch, time_steps, 2, height, width)的数组
    :param max_lag: 自相关的最大滞后步数, 默认为time_steps的一半
    :return: 自相关系数矩阵，形状为(time_steps, max_lag+1)
    """
    batch, time_steps, _, height, width = data.shape
    if max_lag is None:
        max_lag = time_steps // 2
    autocorr = np.zeros((time_steps, max_lag + 1))
    
    # 对每个时间序列的每个速度分量求平均自相关
    for i in range(time_steps):
        for dim in range(2):  # 遍历x,y速度维度
            series = data[:, i, dim, :, :].reshape(batch, -1)  
            autocorr[i, :] = np.correlate(series.mean(axis=0), series.mean(axis=0), mode='full')[:max_lag+1]
            
    return autocorr

def compute_space_autocorrelation(data, max_lag=None):
    """
    计算数据的空间维度自相关 分别针对x和y速度分量。
    
    :param data: 同上
    :param max_lag: 自相关的最大滞后步数 默认为height或width中较小值的一半
    :return: 自相关系数矩阵，形状为(2, time_steps, height, width, max_lag+1)
    """
    batch, time_steps, _, height, width = data.shape
    if max_lag is None:
        max_lag = min(height, width) // 2
    space_autocorr = np.zeros((2, time_steps, height, width, max_lag + 1))
    
    for dim in range(2):  # 针对x和y维度
        for t in range(time_steps):
            for lag in range(max_lag + 1):
                # 分别在高度和宽度上滚动数据
                shifted_height = np.roll(data[:, t, dim, :, :], lag, axis=-2)
                shifted_width = np.roll(data[:, t, dim, :, :], lag, axis=-1)
                
                # 计算自相关并累加到结果中
                space_autocorr[dim, t, :, :, lag] = np.mean((data[:, t, dim, :, :] * shifted_height), axis=(0, -1))  # 高度方向
                space_autocorr[dim, t, :, :, lag] += np.mean((data[:, t, dim, :, :] * shifted_width), axis=(0, -2))  # 宽度方向
                
    # 可选：返回每个时间步的平均自相关（取决于分析需求）
    # return np.mean(space_autocorr, axis=1)
    return space_autocorr  # 保持时间维度以供进一步分析



if __name__ == "__main__":
    # 假设你有如下数据
    generated_samples = torch.rand(8, 40, 256, 256)  # 生成的图像，形状为(N, 40, 256, 256)
    real_samples = torch.rand(8, 40, 256, 256)  # 真实的对应图像，形状需要与生成图像一致

    # 计算MSE
    mse_error = calculate_mse_loss(generated_samples, real_samples)
    print(f"MSE Error: {mse_error.item()}")

    
    # 使用SSIM计算相似度
    ssim_score = ssim(generated_samples, real_samples)
    print(f"SSIM Score: {ssim_score.item()}")


    # 示例数据
    data_gen = np.random.rand(8, 20, 2, 256, 256)  # 示例生成数据
    data_real = np.random.rand(8, 20, 2, 256, 256)  # 示例真实数据

    # 计算时间自相关
    time_auto_gen = compute_time_autocorrelation(data_gen)
    time_auto_real = compute_time_autocorrelation(data_real)

    # 计算空间自相关
    space_auto_gen = compute_space_autocorrelation(data_gen)
    space_auto_real = compute_space_autocorrelation(data_real)

    # 接下来可以根据需要进一步分析生成与真实数据在时空自相关上的差异