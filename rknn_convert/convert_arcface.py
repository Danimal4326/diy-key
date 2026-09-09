import sys
from rknn.api import RKNN

# insightface preprocessing: BGR->RGB, (x - 127.5) / 127.5
MEAN = [[127.5, 127.5, 127.5]]
STD = [[127.5, 127.5, 127.5]]

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} onnx_model_path platform [output_rknn_path]")
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else model_path.replace('.onnx', '.rknn')

    rknn = RKNN(verbose=False)

    print('--> Config model')
    rknn.config(mean_values=MEAN, std_values=STD, target_platform=platform)
    print('done')

    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path, inputs=['input.1'], input_size_list=[[1, 3, 112, 112]])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Embedding model: skip int8 quantization to preserve cosine-similarity
    # precision (unlike detection, there's no NMS/thresholding to absorb error).
    print('--> Building model (fp16, no quantization)')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    rknn.release()
