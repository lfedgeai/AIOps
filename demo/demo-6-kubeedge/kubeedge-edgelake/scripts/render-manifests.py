#!/usr/bin/env python3
"""Create editable environment-specific manifest copies; never apply automatically."""
import argparse,pathlib,re
p=argparse.ArgumentParser()
p.add_argument('--edge-ip',required=True)
p.add_argument('--edge-node',required=True)
p.add_argument('--output',default='generated')
a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[1]
out=root/a.output;out.mkdir(parents=True,exist_ok=True)
for name in ('deployment-edgelake.yaml','deployment-edgelake-dashboard.yaml'):
 s=(root/'manifests/edgelake'/name).read_text()
 s=s.replace('rhel-edge-us',a.edge_node).replace('10.0.0.20',a.edge_ip)
 (out/name).write_text(s)
 print('Rendered',out/name)
print('Review hostPath, imagePullSecrets, env vars, service ports and security before applying.')
