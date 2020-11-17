import onnx
import numpy as np
import onnxruntime as ort



def test():
    # onnx
    ort_sess = ort.InferenceSession('./Hs.onnx')

    with open('1.npy', 'rb') as f:
        input_tensor1 = np.load(f)

    with open('2.npy', 'rb') as f:
        input_tensor2 = np.load(f)
    
    print('=========1.npy============')
    output = ort_sess.run(['output0'], {'input0': input_tensor1,})
    print(output)
    
    print('=========2.npy============')
    output = ort_sess.run(['output0'], {'input0': input_tensor2,})
    print(output)


if __name__ == '__main__':
    test()
    