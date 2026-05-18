import onnx
m = onnx.load('app/models/proctoring.onnx')
print('Proctoring model metadata:')
for p in m.metadata_props:
    print(f'  {p.key}: {p.value}')

m2 = onnx.load('app/models/headset_model.onnx')
print('\nHeadset model metadata:')
for p in m2.metadata_props:
    print(f'  {p.key}: {p.value}')
