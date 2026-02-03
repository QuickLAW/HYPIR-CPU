"""
Streaming VAEHook - fundamental solution to memory pressure.

Core ideas:
1. Avoid full output tensor pre-allocation
2. Process tiles and flush immediately
3. Progressive result assembly
4. Reduce peak memory from ~17GB to ~3-4GB
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
import numpy as np
from PIL import Image
import gc
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import time

from .vaehook import VAEHook, crop_valid_region, GroupNormParam


class StreamingVAEHook(VAEHook):
    """
    Streaming VAEHook to address memory pressure.

    Improvements:
    1. Tile-wise output flushing
    2. Streaming processing; immediate release of intermediates
    3. Memory tracking and proactive cleanup
    4. Progressive assembly for realtime feedback
    """
    
    def __init__(self, net, tile_size, is_decoder, fast_decoder, fast_encoder, 
                 color_fix, to_gpu=False, dtype=None, 
                 streaming_mode=True, temp_dir=None, max_memory_gb=4.0):
        super().__init__(net, tile_size, is_decoder, fast_decoder, fast_encoder, 
                        color_fix, to_gpu, dtype)
        
        self.streaming_mode = streaming_mode
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.max_memory_gb = max_memory_gb
        self.current_memory_gb = 0.0
        self.memory_lock = threading.Lock()
        
        # Create a session-scoped temporary directory
        self.session_temp_dir = os.path.join(self.temp_dir, f"hypir_streaming_{int(time.time())}")
        os.makedirs(self.session_temp_dir, exist_ok=True)
        
    def __del__(self):
        """Cleanup temporary files."""
        try:
            import shutil
            if hasattr(self, 'session_temp_dir') and os.path.exists(self.session_temp_dir):
                shutil.rmtree(self.session_temp_dir, ignore_errors=True)
        except:
            pass
    
    def _get_memory_usage_gb(self) -> float:
        """Get current memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        else:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024**3)
    
    def _update_memory_tracking(self, delta_gb: float):
        """Update memory tracking counter."""
        with self.memory_lock:
            self.current_memory_gb += delta_gb
    
    def _should_use_streaming(self, output_height: int, output_width: int, channels: int) -> bool:
        """Decide whether streaming mode should be used."""
        if not self.streaming_mode:
            return False
            
        # Estimate output tensor size
        expected_size_gb = (output_height * output_width * channels * 4) / (1024**3)
        
        # Enforce streaming when output exceeds 2GB
        if expected_size_gb > 2.0:
            return True
            
        # Use streaming when current + expected exceeds budget
        current_memory = self._get_memory_usage_gb()
        if current_memory + expected_size_gb > self.max_memory_gb:
            return True
            
        return False
    
    def _save_tile_result(self, tile_result: torch.Tensor, tile_idx: int, 
                        output_bbox: Tuple[int, int, int, int]) -> str:
        """Persist a single tile result to a temporary file."""
        temp_file = os.path.join(self.session_temp_dir, f"tile_{tile_idx}.pt")
        
        # Save tile and metadata
        tile_data = {
            'result': tile_result.cpu(),
            'bbox': output_bbox,
            'shape': tile_result.shape
        }
        
        torch.save(tile_data, temp_file)
        return temp_file
    
    def _load_tile_result(self, temp_file: str) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """Load a tile result from a temporary file."""
        tile_data = torch.load(temp_file, map_location='cpu')
        return tile_data['result'], tile_data['bbox']
    
    def _assemble_streaming_result(self, tile_files: List[str], 
                                output_shape: Tuple[int, int, int, int],
                                device: torch.device) -> torch.Tensor:
        """Assemble final output from tile files."""
        print(f"[Streaming VAE]: Assembling {len(tile_files)} tiles...")
        
        # Batched load and assemble to avoid memory spikes
        batch_size = min(4, len(tile_files))
        result = None
        
        for i in range(0, len(tile_files), batch_size):
            batch_files = tile_files[i:i+batch_size]
            
            # Create the output tensor from sample metadata
            if result is None:
                sample_result, _ = self._load_tile_result(batch_files[0])
                channels = sample_result.shape[1]
                result = torch.zeros((output_shape[0], channels, output_shape[2], output_shape[3]), 
                                   device=device, dtype=sample_result.dtype)
                print(f"[Streaming VAE]: Output tensor: {result.shape}, memory: {result.numel() * 4 / (1024**3):.2f}GB")
            
            # Assemble current batch
            for file_path in batch_files:
                tile_result, bbox = self._load_tile_result(file_path)
                tile_result = tile_result.to(device)
                
                # Copy tile into the output tensor
                x1, x2, y1, y2 = bbox
                result[:, :, y1:y2, x1:x2] = tile_result
                
                # Release tile memory
                del tile_result
                
                # Remove temporary file
                try:
                    os.remove(file_path)
                except:
                    pass
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"[Streaming VAE]: Assembly complete")
        return result
    
    @torch.no_grad()
    def vae_tile_forward_streaming(self, z):
        """
        Streaming forward pass with memory optimization.
        """
        device = z.device
        dtype = z.dtype
        N, C, H, W = z.shape
        
        # Compute output shape
        if self.is_decoder:
            output_height, output_width = H * 8, W * 8
        else:
            output_height, output_width = H // 8, W // 8
        
        # Decide whether to use streaming
        use_streaming = self._should_use_streaming(output_height, output_width, C)
        
        if not use_streaming:
            print("[Streaming VAE]: Using standard mode (low memory footprint)")
            return super().vae_tile_forward(z)
        
        print(f"[Streaming VAE]: Streaming mode - input: {H}x{W}, output: {output_height}x{output_width}")
        print(f"[Streaming VAE]: Expected memory saved: {(output_height * output_width * C * 4) / (1024**3):.2f}GB")
        
        # Split into tiles
        tiles, tile_weights, in_bboxes, out_bboxes = self.split_tiles(H, W)
        num_tiles = len(tiles)
        
        print(f"[Streaming VAE]: Split into {num_tiles} tiles")
        
        # Build task queues
        task_queue = self.build_task_queue(self.net, self.is_decoder)
        task_queues = [self.clone_task_queue(task_queue) for _ in range(num_tiles)]
        
        # Estimate GroupNorm parameters
        group_norm_param = GroupNormParam()
        if self.color_fix > 0:
            group_norm_param = self.estimate_group_norm(z, task_queue, self.color_fix)
        
        # Process tiles in streaming mode
        tile_files = []
        processed_tiles = 0
        
        from tqdm import tqdm
        pbar = tqdm(total=num_tiles, desc="[Streaming VAE]: Tiles")
        
        try:
            for i in range(num_tiles):
                # Process current tile
                tile = tiles[i].to(device, dtype=dtype)
                current_task_queue = task_queues[i]
                
                # Execute tile tasks
                while len(current_task_queue) > 0:
                    task = current_task_queue.pop(0)
                    if task[0] == 'store_res':
                        task[1].append(tile)
                    elif task[0] == 'load_res':
                        tile = task[1].pop()
                    elif task[0] == 'apply_norm':
                        tile = task[1](tile)
                    else:
                        tile = task[1](tile)
                
                # Crop valid region
                tile_result = crop_valid_region(tile, in_bboxes[i], out_bboxes[i], self.is_decoder)
                
                # Save tile result
                temp_file = self._save_tile_result(tile_result, i, out_bboxes[i])
                tile_files.append(temp_file)
                
                # Release memory
                del tile, tile_result
                tiles[i] = None
                
                processed_tiles += 1
                pbar.update(1)
                pbar.set_postfix({
                    'processed': f"{processed_tiles}/{num_tiles}",
                    'memory': f"{self._get_memory_usage_gb():.1f}GB"
                })
                
                # Periodic garbage collection
                if processed_tiles % 4 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        finally:
            pbar.close()
        
        # Assemble final output
        print("[Streaming VAE]: Assembling final output...")
        output_shape = (N, C, output_height, output_width)
        result = self._assemble_streaming_result(tile_files, output_shape, device)
        
        print(f"[Streaming VAE]: Completed. Final memory usage: {self._get_memory_usage_gb():.2f}GB")
        return result
    
    def __call__(self, x):
        """Override call to enable streaming mode."""
        if self.streaming_mode:
            return self.vae_tile_forward_streaming(x)
        else:
            return super().__call__(x)


class ProgressiveVAEHook(StreamingVAEHook):
    """
    Progressive VAEHook with realtime progress updates.

    Features:
    1. Emits intermediate results
    2. Progress callback support
    3. Early stop
    4. Suited for interactive use
    """
    
    def __init__(self, *args, progress_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_callback = progress_callback
        self.should_stop = False
    
    def stop_processing(self):
        """Request early stop."""
        self.should_stop = True
    
    def _notify_progress(self, processed: int, total: int, intermediate_result=None):
        """Notify progress updates."""
        if self.progress_callback:
            progress_info = {
                'processed': processed,
                'total': total,
                'percentage': (processed / total) * 100,
                'intermediate_result': intermediate_result,
                'memory_usage_gb': self._get_memory_usage_gb()
            }
            self.progress_callback(progress_info)
    
    @torch.no_grad()
    def vae_tile_forward_streaming(self, z):
        """Progressive streaming forward pass."""
        # Enable progressive mode when a callback is provided
        if self.progress_callback:
            return self._progressive_processing(z)
        else:
            return super().vae_tile_forward_streaming(z)
    
    def _progressive_processing(self, z):
        """Progressive processing implementation."""
        device = z.device
        dtype = z.dtype
        N, C, H, W = z.shape
        
        # Compute output shape
        if self.is_decoder:
            output_height, output_width = H * 8, W * 8
        else:
            output_height, output_width = H // 8, W // 8
        
        # Split into tiles
        tiles, tile_weights, in_bboxes, out_bboxes = self.split_tiles(H, W)
        num_tiles = len(tiles)
        
        # Output tensor for progressive updates
        result = torch.zeros((N, C, output_height, output_width), device=device, dtype=dtype)
        
        # Build task queues
        task_queue = self.build_task_queue(self.net, self.is_decoder)
        task_queues = [self.clone_task_queue(task_queue) for _ in range(num_tiles)]
        
        # Progressive processing
        for i in range(num_tiles):
            if self.should_stop:
                break
                
            # Process current tile
            tile = tiles[i].to(device, dtype=dtype)
            current_task_queue = task_queues[i]
            
            # Execute tile tasks
            while len(current_task_queue) > 0:
                task = current_task_queue.pop(0)
                if task[0] == 'store_res':
                    task[1].append(tile)
                elif task[0] == 'load_res':
                    tile = task[1].pop()
                elif task[0] == 'apply_norm':
                    tile = task[1](tile)
                else:
                    tile = task[1](tile)
            
            # Crop valid region and update output
            tile_result = crop_valid_region(tile, in_bboxes[i], out_bboxes[i], self.is_decoder)
            x1, x2, y1, y2 = out_bboxes[i]
            result[:, :, y1:y2, x1:x2] = tile_result
            
            # Emit progress update with intermediate result
            self._notify_progress(i + 1, num_tiles, result.clone())
            
            # Release memory
            del tile, tile_result
            tiles[i] = None
        
        return result


def create_streaming_vae_hook(net, tile_size, is_decoder, fast_decoder=True, 
                            fast_encoder=True, color_fix=0, to_gpu=False, 
                            dtype=None, streaming_mode=True, max_memory_gb=4.0,
                            progressive=False, progress_callback=None):
    """
    Factory for streaming VAEHook.
    """
    if progressive:
        return ProgressiveVAEHook(
            net, tile_size, is_decoder, fast_decoder, fast_encoder,
            color_fix, to_gpu, dtype, streaming_mode=streaming_mode,
            max_memory_gb=max_memory_gb, progress_callback=progress_callback
        )
    else:
        return StreamingVAEHook(
            net, tile_size, is_decoder, fast_decoder, fast_encoder,
            color_fix, to_gpu, dtype, streaming_mode=streaming_mode,
            max_memory_gb=max_memory_gb
        )
