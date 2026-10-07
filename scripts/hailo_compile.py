import os, argparse
from hailo_sdk_client import ClientRunner

ap = argparse.ArgumentParser()
ap.add_argument("--har", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--out-har", default=None, help="컴파일 후 HAR (profiler용)")
a = ap.parse_args()

runner = ClientRunner(har=a.har)
hef = runner.compile()

os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "wb") as f:
    f.write(hef)
print("SAVED HEF:", a.out, f"({os.path.getsize(a.out)/1e6:.2f} MB)")

if a.out_har:
    runner.save_har(a.out_har)
    print("SAVED HAR:", a.out_har, f"({os.path.getsize(a.out_har)/1e6:.2f} MB)")
