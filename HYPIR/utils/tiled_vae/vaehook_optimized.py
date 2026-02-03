# CPU优化版本的VAEHook - 专门针对第三阶段性能优化
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import multiprocessing as mp
from tqdm import tqdm
import gc

from .vaehook import VAEHook, GroupNormParam, build_task_queue, clone_task_queue, crop_valid_region


class OptimizedVAEHook(VAEHook):
    """
    CPU优化版本的VAEHook，专门针对第三阶段Tiled VAE解码器性能优化
    
    主要优化：
    1. CPU模式下的多线程并行处理
    2. 减少不必要的设备传输
    3. 优化内存使用模式
    4. 改进任务调度策略
    """
    
    def __init__(self, net, tile_size, is_decoder, fast_decoder, fast_encoder, color_fix, to_gpu=False, dtype=None):
        super().__init__(net, tile_size, is_decoder, fast_decoder, fast_encoder, color_fix, to_gpu, dtype)
        
        # CPU优化参数
        self.cpu_workers = min(8, mp.cpu_count())  # 限制最大线程数
        self.device_type = str(next(net.parameters()).device)
        self.is_cpu_mode = self.device_type == "cpu"
        
        # 线程安全锁
        self.norm_lock = Lock()
        self.result_lock = Lock()
        
    def process_tile_cpu_optimized(self, tile_data):
        """
        CPU优化的单tile处理函数
        """
        tile_idx, tile, input_bbox, task_queue = tile_data
        device = next(self.net.parameters()).device
        
        # CPU模式下避免不必要的设备传输
        if self.is_cpu_mode:
            working_tile = tile
        else:
            working_tile = tile.to(device)
        
        group_norm_params = []
        
        # 执行任务队列
        while len(task_queue) > 0:
            task = task_queue.pop(0)
            
            if task[0] == 'pre_norm':
                # 收集GroupNorm参数，稍后统一处理
                group_norm_params.append((working_tile.clone(), task[1]))
                break
            elif task[0] == 'store_res' or task[0] == 'store_res_cpu':
                task_id = 0
                res = task[1](working_tile)
                if not self.fast_mode or task[0] == 'store_res_cpu':
                    if not self.is_cpu_mode:
                        res = res.cpu()
                while task_queue[task_id][0] != 'add_res':
                    task_id += 1
                task_queue[task_id][1] = res
            elif task[0] == 'add_res':
                if self.is_cpu_mode:
                    working_tile += task[1]
                else:
                    working_tile += task[1].to(device)
                task[1] = None
            else:
                working_tile = task[1](working_tile)
        
        return tile_idx, working_tile, input_bbox, task_queue, group_norm_params
    
    @torch.no_grad()
    def vae_tile_forward_optimized(self, z):
        """
        优化版本的VAE tile forward，专门针对CPU性能优化
        """
        device = next(self.net.parameters()).device
        net = self.net
        tile_size = self.tile_size
        is_decoder = self.is_decoder

        z = z.detach()
        N, height, width = z.shape[0], z.shape[2], z.shape[3]
        net.last_z_shape = z.shape

        print(f'[Optimized Tiled VAE]: input_size: {z.shape}, tile_size: {tile_size}, padding: {self.pad}')
        print(f'[Optimized Tiled VAE]: CPU workers: {self.cpu_workers}, Device: {self.device_type}')

        in_bboxes, out_bboxes = self.split_tiles(height, width)

        # 准备tiles
        tiles = []
        for input_bbox in in_bboxes:
            tile = z[:, :, input_bbox[2]:input_bbox[3], input_bbox[0]:input_bbox[1]]
            if not self.is_cpu_mode:
                tile = tile.cpu()
            tiles.append(tile)

        num_tiles = len(tiles)
        
        # 构建任务队列
        single_task_queue = build_task_queue(net, is_decoder)
        
        # Fast mode处理
        if self.fast_mode:
            scale_factor = tile_size / max(height, width)
            z_device = z.to(device) if not self.is_cpu_mode else z
            downsampled_z = F.interpolate(z_device, scale_factor=scale_factor, mode='nearest-exact')
            
            print(f'[Optimized Tiled VAE]: Fast mode enabled, estimating group norm parameters on {downsampled_z.shape[3]} x {downsampled_z.shape[2]} image')
            
            # 分布修正
            std_old, mean_old = torch.std_mean(z_device, dim=[0, 2, 3], keepdim=True)
            std_new, mean_new = torch.std_mean(downsampled_z, dim=[0, 2, 3], keepdim=True)
            downsampled_z = (downsampled_z - mean_new) / std_new * std_old + mean_old
            del std_old, mean_old, std_new, mean_new
            downsampled_z = torch.clamp_(downsampled_z, min=z_device.min(), max=z_device.max())
            
            estimate_task_queue = clone_task_queue(single_task_queue)
            if self.estimate_group_norm(downsampled_z, estimate_task_queue, color_fix=self.color_fix):
                single_task_queue = estimate_task_queue
            del downsampled_z

        task_queues = [clone_task_queue(single_task_queue) for _ in range(num_tiles)]
        del z

        # 初始化结果tensor
        result = None
        
        # CPU并行处理优化
        if self.is_cpu_mode and num_tiles > 1 and self.cpu_workers > 1:
            result = self._parallel_cpu_processing(tiles, in_bboxes, out_bboxes, task_queues, N, height, width, device)
        else:
            result = self._sequential_processing(tiles, in_bboxes, out_bboxes, task_queues, N, height, width, device)
        
        return result
    
    def _parallel_cpu_processing(self, tiles, in_bboxes, out_bboxes, task_queues, N, height, width, device):
        """
        CPU并行处理模式
        """
        num_tiles = len(tiles)
        is_decoder = self.is_decoder
        
        # 初始化结果tensor
        result = torch.zeros(
            (N, tiles[0].shape[1], height * 8 if is_decoder else height // 8, width * 8 if is_decoder else width // 8), 
            device=device, 
            requires_grad=False
        )
        
        # 准备并行处理数据
        tile_data_list = [(i, tiles[i], in_bboxes[i], task_queues[i]) for i in range(num_tiles)]
        
        # 多轮处理，直到所有任务完成
        max_iterations = 10  # 防止无限循环
        iteration = 0
        
        pbar = tqdm(total=num_tiles * len(task_queues[0]), desc="[Optimized Tiled VAE]: Parallel CPU Processing")
        
        while tile_data_list and iteration < max_iterations:
            iteration += 1
            
            # 并行处理当前轮次的tiles
            with ThreadPoolExecutor(max_workers=self.cpu_workers) as executor:
                futures = [executor.submit(self.process_tile_cpu_optimized, tile_data) for tile_data in tile_data_list]
                
                completed_tiles = []
                remaining_tiles = []
                all_group_norm_params = []
                
                for future in as_completed(futures):
                    tile_idx, processed_tile, input_bbox, remaining_queue, group_norm_params = future.result()
                    
                    # 更新进度
                    original_queue_len = len(task_queues[tile_idx])
                    completed_tasks = original_queue_len - len(remaining_queue)
                    pbar.update(completed_tasks)
                    
                    if len(remaining_queue) == 0:
                        # tile处理完成
                        completed_tiles.append((tile_idx, processed_tile, input_bbox))
                    else:
                        # tile还有剩余任务
                        remaining_tiles.append((tile_idx, processed_tile, input_bbox, remaining_queue))
                    
                    # 收集GroupNorm参数
                    all_group_norm_params.extend(group_norm_params)
            
            # 处理完成的tiles
            for tile_idx, processed_tile, input_bbox in completed_tiles:
                out_bbox = out_bboxes[tile_idx]
                result[:, :, out_bbox[2]:out_bbox[3], out_bbox[0]:out_bbox[1]] = crop_valid_region(
                    processed_tile, input_bbox, out_bbox, is_decoder
                )
            
            # 处理GroupNorm参数（如果有的话）
            if all_group_norm_params:
                group_norm_param = GroupNormParam()
                for tile, layer in all_group_norm_params:
                    group_norm_param.add_tile(tile, layer)
                
                group_norm_func = group_norm_param.summary()
                if group_norm_func is not None:
                    # 为剩余的task queues添加group norm任务
                    for tile_idx, processed_tile, input_bbox, remaining_queue in remaining_tiles:
                        remaining_queue.insert(0, ('apply_norm', group_norm_func))
            
            # 更新下一轮处理的数据
            tile_data_list = remaining_tiles
            
            # 清理内存
            gc.collect()
        
        pbar.close()
        return result
    
    def _sequential_processing(self, tiles, in_bboxes, out_bboxes, task_queues, N, height, width, device):
        """
        传统串行处理模式（作为fallback）
        """
        # 使用原始的串行处理逻辑，但进行一些CPU优化
        return super().vae_tile_forward(torch.cat(tiles, dim=0).reshape(N, -1, height, width))


# 工厂函数，根据设备类型选择合适的VAEHook
def create_optimized_vae_hook(net, tile_size, is_decoder, fast_decoder, fast_encoder, color_fix, to_gpu=False, dtype=None):
    """
    根据设备类型创建优化的VAEHook
    """
    device_type = str(next(net.parameters()).device)
    
    if device_type == "cpu":
        return OptimizedVAEHook(net, tile_size, is_decoder, fast_decoder, fast_encoder, color_fix, to_gpu, dtype)
    else:
        # GPU模式使用原始实现
        return VAEHook(net, tile_size, is_decoder, fast_decoder, fast_encoder, color_fix, to_gpu, dtype)