import tensorrt as trt
import argparse

parsers = argparse.ArgumentParser(description = "onnx lanenet")
parsers.add_argument("--onnx_dir",required=True, help = "need onnx lanenet_model")
parsers.add_argument("--trt_dir",required=True, help = "need tensorRT engine")
args = parsers.parse_args()

onnx_file_name = args.onnx_dir

def ONNX_build_engine(onnx_file_path, engine_file_path):
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(G_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, G_LOGGER) as parser:
        builder.max_batch_size = 1
        builder.max_workspace_size = 1 << 30

        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            parser.parse(model.read())
        print("network.num_layers : {}".format(network.num_layers))
        last_layer = network.get_layer(network.num_layers - 1)
        print("Check if last layer recognizes it's output")
        if not last_layer.get_output(0):
            # If not, then mark the output using TensorRT API
            network.mark_output(last_layer.get_output(0))
        print('Completed parsing of ONNX file')

        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        engine = builder.build_cuda_engine(network)
        print("Completed creating Engine")

        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        return engine

if __name__ == "__main__":
    ONNX_build_engine(onnx_file_name, args.trt_dir+'/Lanenet.trt')