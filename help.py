import onnx
model = onnx.load("best_model.onnx")
print(onnx.helper.printable_graph(model.graph))
