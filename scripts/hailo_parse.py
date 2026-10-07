import os, argparse
from hailo_sdk_client import ClientRunner

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True)
ap.add_argument("--end-nodes-file", required=True)
ap.add_argument("--name", default="yolo11n_640")
ap.add_argument("--hw-arch", default="hailo10h")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--out", required=True)
a = ap.parse_args()

ends = [l.strip() for l in open(a.end_nodes_file) if l.strip() and not l.startswith("#")]
print(f"end nodes ({len(ends)}):")
for e in ends:
    print("  ", e)

runner = ClientRunner(hw_arch=a.hw_arch)
runner.translate_onnx_model(
    a.onnx,
    a.name,
    start_node_names=["images"],
    end_node_names=ends,
    net_input_shapes={"images": [1, 3, a.imgsz, a.imgsz]},
)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
runner.save_har(a.out)
print("\nSAVED:", a.out, f"({os.path.getsize(a.out)/1e6:.2f} MB)")
