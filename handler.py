"""
SDXL Turbo Worker for RunPod Serverless
Ultra-fast image generation with Stable Diffusion XL Turbo
"""

import os
import base64
import io
import time
from typing import Optional, Dict, Any
import torch.nn as nn
import kornia.geometry.transform as KT

import torch
import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from diffusers import AutoPipelineForText2Image
from PIL import Image

from schemas import INPUT_SCHEMA


class ModelHandler:
    def __init__(self):
        """Initialize the SDXL Turbo pipeline."""
        self.pipe = None
        self.load_model()

    def load_model(self):
        """Load the SDXL Turbo model."""
        print("🚀 Loading SDXL Turbo model...")

        try:
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
                local_files_only=False,
            )

            if torch.cuda.is_available():
                self.pipe.to("cuda")
                print("✅ Model loaded successfully on GPU!")
            else:
                print("⚠️  GPU not available, running on CPU")

        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load SDXL Turbo model: {str(e)}")

    def generate_image(self, job_input: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image using SDXL Turbo."""

        # Extract parameters
        prompt = job_input.get("prompt")
        bending_layer = job_input.get("bending_layer", "")
        bending_param = job_input.get("bending_param", 0)
        negative_prompt = job_input.get("negative_prompt")
        height = job_input.get("height", 512)
        width = job_input.get("width", 512)
        num_inference_steps = job_input.get("num_inference_steps", 1)
        guidance_scale = job_input.get("guidance_scale", 0.0)
        num_images = job_input.get("num_images", 1)
        seed = job_input.get("seed")

        print(f"🎨 Generating {num_images} image(s) with prompt: '{prompt[:50]}...'")
        print(
            f"📐 Size: {width}x{height}, Steps: {num_inference_steps}, Guidance: {guidance_scale}"
        )

        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        try:
            start_time = time.time()

            bend = RotateModule(bending_param)
            handle = hook_module(self.pipe.unet, bending_layer, bend)

            # Generate images
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
            )
            unhook_module(self.pipe.unet, bending_layer, handle)

            generation_time = time.time() - start_time
            print(f"⚡ Generated in {generation_time:.2f} seconds")

            # Process images
            images_data = []
            for i, image in enumerate(result.images):
                # Convert to base64
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")

                images_data.append(
                    {"image": image_b64, "seed": seed + i if seed is not None else None}
                )

            return {
                "images": images_data,
                "generation_time": generation_time,
                "parameters": {
                    "prompt": prompt,
                    "bending_layer": bending_layer,
                    "bending_param": bending_param,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed,
                },
            }

        except Exception as e:
            print(f"❌ Error during generation: {str(e)}")
            raise RuntimeError(f"Image generation failed: {str(e)}")


# Initialize model handler
model_handler = ModelHandler()


def handler(job):
    """
    Handler function for RunPod serverless.
    """
    try:
        # Validate input
        job_input = job["input"]

        # Validate against schema
        validated_input = validate(job_input, INPUT_SCHEMA)
        if "errors" in validated_input:
            return {"error": f"Input validation failed: {validated_input['errors']}"}

        validated_data = validated_input["validated_input"]

        # Generate image
        result = model_handler.generate_image(validated_data)

        return result

    except Exception as e:
        print(f"❌ Handler error: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    print("🎯 Starting SDXL Turbo Worker...")
    runpod.serverless.start({"handler": handler})




class BendingModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        # Check if the input is 3D (single image), and add a batch dimension if necessary.
        num_unsqueeze_added = 0
        if x.ndim == 2:  # Single channel image: (H, W)
            x = x.unsqueeze(0).unsqueeze(0)  # Convert to (1, 1, H, W)
            num_unsqueeze_added = 2
        if x.ndim == 3:  # Single image: (C, H, W)
            x = x.unsqueeze(0)  # Convert to (1, C, H, W)
            num_unsqueeze_added = 1
        elif x.ndim != 4:  # Expect a 4D tensor: (B, C, H, W)
            raise ValueError(
                f"Input tensor must be 3D or 4D, but got ndim={x.ndim}")

        # Delegate the transformation to the child class using a separate method.
        output = self.bend(x, *args, **kwargs)

        # If we added a batch dimension, remove it from the output.
        for i in range(num_unsqueeze_added):
            output = output.squeeze(0)
        return output

    def bend(self, x, *args, **kwargs):
        # Child classes must override this method
        raise NotImplementedError(
            "Subclasses must implement the compute() method.")


class RotateModule(BendingModule):
    def rotate_image(self, degrees):
        def fn(x):
            B = x.shape[0]
            angle_tensor = torch.full(
                (B,), degrees, device=x.device, dtype=x.dtype)
            return KT.rotate(x, angle=angle_tensor)
        return fn

    def __init__(self, angle_degrees=0):
        super().__init__()
        self.angle_degrees = angle_degrees

    def bend(self, x, *args, **kwargs):
        print("rotating now!")
        return self.rotate_image(self.angle_degrees)(x)


def hook_module(model: nn.Module, layer_path: str, new_module: nn.Module):
    """
    Injects new_module into model at the specified layer_path.

    - If the target module supports hooks, a forward hook is registered that calls new_module,
      passing output, positional arguments, and keyword arguments.
    - If the target module is not hookable (e.g. a bare nn.ModuleList which has no proper forward),
      the target is wrapped in a new container (a Sequential) that appends new_module.

    Args:
        model (nn.Module): The model instance.
        layer_path (str): Dot-separated path to the target module
            e.g. "features.3" or "block.0"
        new_module (nn.Module): The module to inject.

    Returns:
        A hook handle if a hook is registered, otherwise None.
    """
    if not layer_path:
        return

    # Split and navigate to the parent of the target module.
    parts = layer_path.split('.')
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    last_part = parts[-1]

    # Get the target module reference.
    if last_part.isdigit():
        idx = int(last_part)
        target_module = parent[idx]
    else:
        target_module = getattr(parent, last_part)

    # Decide whether to hook or wrap:
    # In this example, if the target is an nn.ModuleList (that isn’t already a Sequential),
    # we consider it not properly hookable.
    if isinstance(target_module, nn.ModuleList):
        print("Module List detected, wrapping in Sequential.")
        # Wrap by creating a new Sequential that contains the modules from target_module,
        # and then appending the new_module.
        # We don't simply append to the ModuleList instance because there is no guarantee that the list's items will be iterated on and called during the model execution.
        modules = list(target_module)
        modules.append(new_module)
        new_seq = nn.Sequential(*modules)
        # Replace the attribute in the parent with the new Sequential.
        if last_part.isdigit():
            parent[int(last_part)] = new_seq
        else:
            setattr(parent, last_part, new_seq)
        print(
            f"Wrapped ModuleList at '{layer_path}' in a Sequential that appends the new module.")
        return None
    elif isinstance(target_module, nn.Sequential):
        target_module.append(new_module)
    else:
        # Otherwise, register a forward hook.
        def hook(module, args, kwargs, output):
            # Call the injected new_module, passing the output plus original args and kwargs.
            return new_module(output, *args, **kwargs)

        # The with_kwargs flag lets the hook capture both positional args and keyword args.
        handle = target_module.register_forward_hook(hook, with_kwargs=True)
        print(target_module, dir(target_module), handle)
        print(f"Registered hook on module at '{layer_path}'.")
        return handle


def unhook_module(model: nn.Module, layer_path: str, hook_handle=None):
    """
    Reverses the effect of `hook_module`:
    - If a hook handle was returned by hook_module, it removes the hook.
    - If the target layer was wrapped (Sequential replacing a ModuleList),
      it unwraps it back to the original ModuleList.
    - If the target layer is a Sequential and a module was appended,
      it removes the last appended module.

    Args:
        model (nn.Module): The model instance.
        layer_path (str): Dot-separated path to the target module (same as used in hook_module).
        hook_handle (torch.utils.hooks.RemovableHandle or None): Optional handle returned by hook_module.
    """
    if not layer_path:
        return

    # Split and navigate to the parent of the target module
    parts = layer_path.split('.')
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    last_part = parts[-1]

    # Get the target module reference
    if last_part.isdigit():
        idx = int(last_part)
        target_module = parent[idx]
    else:
        target_module = getattr(parent, last_part)

    # --- CASE 1: If we have a hook handle, remove it ---
    if hook_handle is not None:
        hook_handle.remove()
        print(f"Removed forward hook from '{layer_path}'.")
        return

    # --- CASE 2: If the target was wrapped Sequential (originally a ModuleList) ---
    if isinstance(target_module, nn.Sequential):
        # Check if the original module was likely a wrapped ModuleList
        # by verifying that the original Sequential’s submodules correspond to a
        # previous ModuleList + injected new module.
        # We can’t be 100% sure, so we check for a message pattern (last one likely injected)
        # or that the parent path still has a record.
        if len(target_module) > 0:
            # Remove last appended module
            modules = list(target_module.children())[:-1]
            # Try to infer original type (ModuleList)
            restored = nn.ModuleList(modules)
            if last_part.isdigit():
                parent[int(last_part)] = restored
            else:
                setattr(parent, last_part, restored)
            print(
                f"Reverted Sequential at '{layer_path}' back to ModuleList (removed appended module).")
            return

    # --- CASE 3: If the target is a Sequential and we only appended a module ---
    elif isinstance(target_module, nn.Sequential):
        if len(target_module) > 0:
            removed_module = target_module[-1]
            del target_module[-1]
            print(
                f"Removed appended module '{removed_module.__class__.__name__}' from Sequential at '{layer_path}'.")
        return

    print(f"No modifications detected at '{layer_path}' to undo.")
