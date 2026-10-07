import sys, re, onnx

g = onnx.load(sys.argv[1]).graph
idx = max(int(x) for n in g.node for x in re.findall(r"/model\.(\d+)/", n.name or ""))
pref = f"/model.{idx}/"

consumers = {}
for n in g.node:
    for i in n.input:
        consumers.setdefault(i, []).append(n)

ends = []
for n in g.node:
    nm = n.name or ""
    # cv2.N(박스) / cv3.N(클래스) 브랜치의 마지막 Conv만. dfl은 디코드 경로이므로 제외
    if n.op_type == "Conv" and nm.startswith(pref) and re.search(r"/cv[23]\.\d+/", nm):
        for c in consumers.get(n.output[0], []):
            if c.op_type in ("Concat", "Reshape"):
                ends.append(n.name)
                break

print(f"# detect module {pref} / end nodes {len(ends)}개", file=sys.stderr)
for e in ends:
    print(e)
