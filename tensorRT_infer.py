import time
import os
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import timeit
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="video path")
    parser.add_argument("--model", help="Model path", default='./save/Lanenet.trt')
    return parser.parse_args()


TRT_LOGGER = trt.Logger()
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(args.video, fourcc, 30, (512,256))

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def engines_create(engine, input_file, image_height, image_width):
    print("Reading input image from file {}".format(input_file))
    data_transform = transforms.Compose([
        transforms.Resize((image_height,  image_width)),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    with engine.create_execution_context() as context:
        context.set_binding_shape(engine.get_binding_index("input"), (1, 3, image_height, image_width))


        cap = cv2.VideoCapture(input_file)
        while(cap.isOpened()):

            ret, frame = cap.read()
            if ret:
                dummy_input = Image.fromarray(frame)
                dummy_input = data_transform(dummy_input)
                input_image = torch.unsqueeze(dummy_input, dim=0).numpy()
                start_t = timeit.default_timer()
                bindings = []

                for binding in engine:

                    binding_idx = engine.get_binding_index(binding)
                    size = trt.volume(context.get_binding_shape(binding_idx))
                    dtype = trt.nptype(engine.get_binding_dtype(binding))

                    if engine.binding_is_input(binding):
                        input_buffer = np.ascontiguousarray(input_image)
                        input_memory = cuda.mem_alloc(input_image.nbytes)
                        bindings.append(int(input_memory))
                    else:
                        output_buffer = cuda.pagelocked_empty(size, dtype)
                        output_memory = cuda.mem_alloc(output_buffer.nbytes)
                        bindings.append(int(output_memory))

                stream = cuda.Stream()
                cuda.memcpy_htod_async(input_memory, input_buffer, stream)
                context.execute_async(bindings=bindings, stream_handle=stream.handle)
                cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
                terminate_t = timeit.default_timer()
                stream.synchronize()
                FPS = int(1./(terminate_t - start_t ))
                print("FPS : ",FPS)
                out_img = output_buffer.reshape(3,256,512)*255
                out_img = out_img.transpose((1,2,0))
                out.write(out_img.astype(np.uint8))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        out.release()
        cap.release()
        cv2.destroyAllWindows()

def test():
    args = parse_args()
    with load_engine(args.model) as engine:
        img_path = args.img
        if img_path is None:
            return False
        image_height = args.height
        image_width = args.width
        engines_create(engine, img_path, image_height, image_width )
    return True

if __name__ == "__main__":
    if test():
        print("end_trt_engine")
