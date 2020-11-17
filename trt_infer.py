import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import common
import time



def test():
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open("./Hs.engine", "rb") as f:
        with trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            inputs, outputs, bindings, stream = common.allocate_buffers(engine, [1*200*832]*2)
            context = engine.create_execution_context()

            with open('1.npy', 'rb') as f:
                input_tensor1 = np.load(f)

            with open('2.npy', 'rb') as f:
                input_tensor2 = np.load(f)

            inputs[0].host = input_tensor1
           
            tic = time.time()
            trt_outputs = common.do_inference_hs(context,
                    bindings=bindings,
                    inputs=inputs,
                    outputs=outputs,
                    stream=stream,
                    t1=1,
                    t2=200,
                    t3=832)
            print(trt_outputs)
            print('=========1.npy============')
            output = trt_outputs
            output = output[0].reshape((1, 1, 200, 832)) # 1.npy
            print(output)

            inputs[0].host = input_tensor2
           
            tic = time.time()
            trt_outputs = common.do_inference_hs(context,
                    bindings=bindings,
                    inputs=inputs,
                    outputs=outputs,
                    stream=stream,
                    t1=1,
                    t2=200,
                    t3=832)
            print(trt_outputs)
            print('=========2.npy============')
            output = trt_outputs
            output = output[0].reshape((1, 1, 832, 200)).transpose((0, 1, 3, 2)) # 2.npy
            print(output)


if __name__ == '__main__':
    test()
    




