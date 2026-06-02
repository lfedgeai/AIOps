eyJhbGciOiJSUzI1NiIsImtpZCI6IkFPTGJNdUZqOHYyUXZVRVRMWUdoMkYzWm1DQmQ0eEs2N19DWmRUdlp4Zm8ifQ.eyJhdWQiOlsiaHR0cHM6Ly9rdWJlcm5ldGVzLmRlZmF1bHQuc3ZjLmNsdXN0ZXIubG9jYWwiLCJrM3MiXSwiZXhwIjoxNzgwMzQ5MDU0LCJpYXQiOjE3ODAzNDU0NTQsImlzcyI6Imh0dHBzOi8va3ViZXJuZXRlcy5kZWZhdWx0LnN2Yy5jbHVzdGVyLmxvY2FsIiwianRpIjoiZjAzNzBlZTMtZDI3NS00ZTA1LTk5OGMtZjk4OTljMzhhODc2Iiwia3ViZXJuZXRlcy5pbyI6eyJuYW1lc3BhY2UiOiJrdWJlcm5ldGVzLWRhc2hib2FyZCIsInNlcnZpY2VhY2NvdW50Ijp7Im5hbWUiOiJhZG1pbi11c2VyIiwidWlkIjoiZTcwMmU3MjgtZjA1ZS00OTI0LTk1NzUtYTBkZDdiZmI3ZTVmIn19LCJuYmYiOjE3ODAzNDU0NTQsInN1YiI6InN5c3RlbTpzZXJ2aWNlYWNjb3VudDprdWJlcm5ldGVzLWRhc2hib2FyZDphZG1pbi11c2VyIn0.OVJtpEUdX54Z5IG5Tuy1VLtfOS3eApBTS9o7ZXvNsMjMbVrjQu8Ddq0qyvUN2KUDn7rb878jwz-i5KHd5ZX5qi0Z-QgDLfiPE56LPln59C6dVq1VmWgqDkSUdfI2TK8qSv-LVEvp-6gnQ3ZFi1Br-gfsB_xnBdLsNSbpe-TboevkIU35ae3PsZB_1U_eVjs5yUF_xDPw0L0ZH6RBzviqVrwt9lowumBlch8xiJAOHjCeYkapOQ-wrUrCa8-148ewomGNK7kcc4GL-m6VVWuWL_HrWihW4UTTXSxKksarW8fDelGCOGKRYmDNtuRKbNiG-z6xP5C-2wJKb2S5Xzy4fQ


kubectl -n kubernetes-dashboard create token admin-user

Generate a long-lived token

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: admin-user-token
  namespace: kubernetes-dashboard
  annotations:
    kubernetes.io/service-account.name: admin-user
type: kubernetes.io/service-account-token
EOF

kubectl -n kubernetes-dashboard describe secret admin-user-token

//long live token
eyJhbGciOiJSUzI1NiIsImtpZCI6IkFPTGJNdUZqOHYyUXZVRVRMWUdoMkYzWm1DQmQ0eEs2N19DWmRUdlp4Zm8ifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJrdWJlcm5ldGVzLWRhc2hib2FyZCIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VjcmV0Lm5hbWUiOiJhZG1pbi11c2VyLXRva2VuIiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZXJ2aWNlLWFjY291bnQubmFtZSI6ImFkbWluLXVzZXIiLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC51aWQiOiJlNzAyZTcyOC1mMDVlLTQ5MjQtOTU3NS1hMGRkN2JmYjdlNWYiLCJzdWIiOiJzeXN0ZW06c2VydmljZWFjY291bnQ6a3ViZXJuZXRlcy1kYXNoYm9hcmQ6YWRtaW4tdXNlciJ9.MszcbO0thAl2jgMcCBkKuObsSp5hF9aFrHW-QGCMYOlKkV-Vqtdrnk5f2bvi_SN9kK-LKFO4UGX8dwGi7gz7MYBWAENyMWFlPuXrURs3swCd75nvZZDQFahKySNE4yw4y2HGk_TdfQv4MiUX6yI750E-T2mTcUVr_I3Pf7989qmRecOPCpSnr7-cSrGm4rhhoRMAnorCrokdIyZxg1iNcZ1S0k18tnIbtcuNQMwsU4gn8_mSS0tq2VLxpTYYysiaDhSNzFB7zVbn_oAt0US_8Fj_otxfqlGKUnUAo9qaMyhefVKJUvL31DLGr3uP4ErER20iHmpQ8vCCsJJy7KNgCw


To avoid typing port-forward every time, create a small script:

cat <<'EOF' > start-dashboard.sh
#!/bin/bash
kubectl -n kubernetes-dashboard port-forward svc/kubernetes-dashboard 8443:443 --address 0.0.0.0
EOF

chmod +x start-dashboard.sh
./start-dashboard.sh

kubectl -n kubernetes-dashboard port-forward svc/kubernetes-dashboard 8443:443 --address 0.0.0.0

// keadm install

sudo keadm init \
  --advertise-address="192.168.1.76" \
  --kube-config=/etc/rancher/k3s/k3s.yaml

kubectl get pods -n kubeedge

token
sudo keadm gettoken --kube-config=/etc/rancher/k3s/k3s.yaml

913277f59e6f8e3bebd453f159018c577452bec991cbb3470de3b0e0a48087a3.eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3ODA0MzI3NzV9.4hW8kZfS2N9VQfmksx_7JCK4hE9ALISsCpjazGwUFD8

//ocp server machin
https://192.168.1.112:9090/

<!-- sudo keadm join \
  --cloudcore-ipport=192.168.1.76:10000 \
  --token=913277f59e6f8e3bebd453f159018c577452bec991cbb3470de3b0e0a48087a3.eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3ODA0MzI3NzV9.4hW8kZfS2N9VQfmksx_7JCK4hE9ALISsCpjazGwUFD8
 -->

sudo /usr/local/bin/keadm join \
  --cloudcore-ipport=192.168.1.76:10000 \
  --token=913277f59e6f8e3bebd453f159018c577452bec991cbb3470de3b0e0a48087a3.eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3ODA0MzI3NzV9.4hW8kZfS2N9VQfmksx_7JCK4hE9ALISsCpjazGwUFD8


  nohup ./start-dashboard.sh > dashboard.log 2>&1 &

  ps -ef | grep port-forward

to kill proce 
  pkill -f "kubectl -n kubernetes-dashboard port-forward"

graphan
  http://192.168.1.76:30090

promethius 
http://192.168.1.76:30091/targets

metric 
http://192.168.1.76:8889/metrics

  system_cpu_load_average_1m

  system_cpu_time_seconds_total
  system_memory_usage_bytes
  system_network_io_bytes_total


   kubectl get nodes